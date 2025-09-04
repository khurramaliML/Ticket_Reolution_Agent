import json
import os
from typing import Dict, List, Optional, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

MODELID = "moonshotai/kimi-k2-instruct"

class TicketCategory(Enum):
    BILLING = "billing"
    TECHNICAL_ISSUE = "technical_issue"
    SECURITY = "security"
    GENERAL_INQUIRY = "general_inquiry"

class AgentState(TypedDict):
    ticket_subject: str
    ticket_description: str
    category: Optional[TicketCategory]
    context: List[str]
    draft_response: Optional[str]
    review_feedback: Optional[str]
    is_approved: bool
    retry_count: int
    max_retries: int
    final_response: Optional[str]
    conversation_history: List[BaseMessage]

def classify_ticket(state: AgentState):
    """Classify the ticket into category using LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a ticket classification expert. Analyze the ticket and classify it.

        Available categories: {categories}

        Return a JSON response with 'category' field."""),
        ("human", "categories: {categories}\nSubject: {subject}\nDescription: {description}")
    ])

    categories = [cat.value for cat in TicketCategory]
    model = ChatGroq(
        model=MODELID,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    chain = prompt | model 

    response = chain.invoke({
        "categories": categories,
        "subject": state["ticket_subject"],
        "description": state["ticket_description"]
    })

    json_response = json.loads(response.content)
    state["category"] = TicketCategory(json_response.get("category", TicketCategory.GENERAL_INQUIRY))
    return state

def retrieve_context(state: AgentState) -> AgentState:
    """Fetch contextual information based on ticket category."""
    # Mocked knowledge sources per category
    knowledge_sources = {
        TicketCategory.BILLING: [
            "Billing FAQ: Refunds are processed within 5 business days.",
            "Contact billing@company.com for invoice issues."
        ],
        TicketCategory.TECHNICAL_ISSUE: [
            "Technical troubleshooting guide: Restart your device.",
            "Known issues: Service outage on 2024-06-01."
        ],
        TicketCategory.SECURITY: [
            "Security protocol: Change passwords every 90 days.",
            "Report suspicious activity to security@company.com."
        ],
        TicketCategory.GENERAL_INQUIRY: [
            "General inquiries: Visit our help center for more info.",
            "Contact support@company.com for further assistance."
        ]
    }

    category = state["category"]
    subject = state["ticket_subject"]
    description = state["ticket_description"]

    # Simulate retrieval by filtering relevant docs/snippets
    docs = []
    if category in knowledge_sources:
        # Incorporate subject/description in a simple keyword match (mocked)
        for doc in knowledge_sources[category]:
            if subject.lower() in doc.lower() or description.lower() in doc.lower():
                docs.append(doc)
        # Modular routing: log which category's knowledge source is being queried
        print(f"Retrieving context from knowledge source for category: {category.value}")
        if not docs:
            docs = knowledge_sources[category]
    else:
        docs = ["No relevant context found."]

    state["context"] = docs
    return state

def draft_response(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support agent. Draft a professional, empathetic, and helpful response to the customer's ticket.
        Use the provided context to inform your response. Be specific and actionable.
        Context: {context}
        Guidelines:
        - Be empathetic and understanding
        - Provide clear, actionable steps
        - Include relevant information from context
        - Maintain a professional tone
        - If you can't resolve the issue, explain next steps
        """),
        ("human", """Ticket Details:\nSubject: {subject}\nDescription: {description}\nCategory: {category}\nPlease draft a response to this customer.""")
    ])
    model = ChatGroq(
        model=MODELID,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    chain = prompt | model
    response = chain.invoke({
        "context": "\n".join(state["context"]),
        "subject": state["ticket_subject"],
        "description": state["ticket_description"],
        "category": state["category"].value if state["category"] else "general_inquiry",
    })
    state["draft_response"] = response.content
    return state
from pydantic import BaseModel, Field
class ReviewFeedback(BaseModel):
    approved: bool
    comments: Optional[str]

def review_response(state: AgentState) -> AgentState:
    """Review the draft response for accuracy, helpfulness, and policy compliance."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a customer support quality assurance reviewer. 
        Evaluate the draft response for accuracy, helpfulness, and compliance with support guidelines:
        Guidelines:
        - Be empathetic and understanding
        - Provide clear, actionable steps
        - Include relevant information from context
        - Maintain a professional tone
        - If you can't resolve the issue, explain next steps

        Return a JSON with:
        - 'approved': true/false
        - 'feedback': If rejected, provide specific feedback for revision.
        """),
        ("human", """
        Ticket Details:
        Subject: {subject}
        Description: {description}
        Category: {category}
        Draft Response: {draft_response}
        """)
    ])
    model = ChatGroq(
        model=MODELID,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )
    model = model.bind_tools([ReviewFeedback])
    chain = prompt | model 
    response = chain.invoke({
        "subject": state["ticket_subject"],
        "description": state["ticket_description"],
        "category": state["category"].value if state["category"] else "general_inquiry",
        "draft_response": state["draft_response"],
    })
    result = response.tool_calls[0]["args"]
    state["is_approved"] = result.get("approved", False)
    state["review_feedback"] = result.get("feedback", None)
    return state

initial_state = AgentState(
    ticket_subject="Billing issue",
    ticket_description="I was overcharged this month.",
    category=None,
    context=[],
    draft_response=None,
    review_feedback=None,
    is_approved=False,
    retry_count=0,
    max_retries=2
)

def should_retry(state: AgentState) -> str:
    """Determine if we should retry, approve, or stop due to max retries"""
    if state["is_approved"]:
        return "approve"
    elif state["retry_count"] >= state["max_retries"]:
        return "max_retries"
    else:
        return "retry"
    
def refine_context(state: AgentState) -> AgentState:
    """Refine context based on review feedback"""
    state["retry_count"] += 1
        
    # Add additional context based on feedback
    feedback_keywords = state["review_feedback"].lower().split()
        
    additional_context = []
        
    if "specific" in feedback_keywords or "detail" in feedback_keywords:
        additional_context.append("Provide more specific technical details and step-by-step instructions")
        
    if "empathy" in feedback_keywords or "tone" in feedback_keywords:
        additional_context.append("Use more empathetic language and acknowledge customer frustration")

    if "action" in feedback_keywords or "steps" in feedback_keywords:
        additional_context.append("Include clear, numbered action steps for the customer to follow")
        
    state["context"].extend(additional_context)
        
    state["conversation_history"].append(
        HumanMessage(content=f"Refined context (attempt {state['retry_count']}): Added {len(additional_context)} items")
    )
        
    return state

def finalize_response(state: AgentState) -> AgentState:
    """Finalize the response, even if not approved after max retries"""
    if state["is_approved"]:
        state["final_response"] = state["draft_response"]
    else:
        # If max retries reached, add a disclaimer
        state["final_response"] = f"{state['draft_response']}\n\n[Note: This response was automatically generated. If you need further assistance, please don't hesitate to contact us.]"
        
    state["conversation_history"].append(
        AIMessage(content="Response finalized and ready for delivery")
    )
        
    return state

# Build workflow
builder = StateGraph(AgentState)
# Add nodes
builder.add_node("classify_ticket", classify_ticket)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("draft_response", draft_response)
builder.add_node("review_response", review_response)
builder.add_node("refine_context", refine_context)
builder.add_node("finalize_response", finalize_response)
        
# Define the flow
builder.set_entry_point("classify_ticket")
        
builder.add_edge("classify_ticket", "retrieve_context")
builder.add_edge("retrieve_context", "draft_response")
builder.add_edge("draft_response", "review_response")
        
# Conditional edges for review outcome
builder.add_conditional_edges(
        "review_response",
        should_retry,
        {
            "retry": "refine_context",
            "approve": "finalize_response",
            "max_retries": "finalize_response"
        }
        )
        
builder.add_edge("refine_context", "draft_response")
builder.add_edge("finalize_response", END)
    
graph = builder.compile()

initial_state = AgentState(
    ticket_subject="Billing issue",
    ticket_description="I was overcharged this month.",
    category=None,
    context=[],
    draft_response=None,
    review_feedback=None,
    is_approved=False,
    retry_count=0,
    max_retries=2,
    final_response=None,
    conversation_history=[]
)
# Execute the workflow
final_state = graph.invoke(initial_state)
print("=== TICKET PROCESSING RESULT ===")
print(f"Subject: {final_state['ticket_subject']}")
print(f"Description: {final_state['ticket_description']}")
print(f"Category: {final_state['category']}")
print(f"Context: {final_state['context']}")
print(f"Draft Response: {final_state['draft_response']}")
print(f"Retries: {final_state['retry_count']}")
print(f"Approved: {final_state['is_approved']}")
print("\n=== FINAL RESPONSE ===")
print(final_state['final_response'])