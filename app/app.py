import streamlit as st
import time
import uuid
import logging

from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)

from logs_elastic import setup_es_handler


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

es_handler = setup_es_handler()

logger = logging.getLogger(__name__)
logger.addHandler(es_handler)


def main():
    logging.info("Starting the olx helpdesk assistant application")
    st.title("OLX helpdesk assistant")

    # Session state initialization
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        logging.info(
            f"New conversation started with ID: {st.session_state.conversation_id}"
        )
    if "count" not in st.session_state:
        st.session_state.count = 0
        logging.info("Feedback count initialized to 0")

    st.write("Welcome to the OLX helpdesk assistant.")

    # Temperature selection. 0 should be labeled as conservative and 1 as risky.
    st.write("Select the creativity level of the answer:")
    temperature = st.slider(
        "Select creativity level (temperature):",
        0.0,
        1.0,
        0.1,
        0.1,
        help="Use 0 for a safe answer and 1 if you have a non-standard question. You risk getting an ungeniune or even ungrammatical answer.",
    )

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        st.session_state.conversation_id = str(uuid.uuid4())
        logging.info(
            f"New conversation started with ID: {st.session_state.conversation_id}"
        )
        logging.info(f"User asked: '{user_input}'")
        with st.spinner("Processing..."):
            start_time = time.time()
            answer_data = get_answer(user_input, temperature)
            end_time = time.time()
            logging.info(f"Answer received in {end_time - start_time:.2f} seconds")
            st.success("Completed!")
            st.write(answer_data["answer"])
            logging.info("Saving conversation to database")
            save_conversation(st.session_state.conversation_id, user_input, answer_data)
            logging.info("Conversation saved successfully")
    # Feedback buttons
    st.subheader("Please share your feedback")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            st.session_state.count += 1
            logging.info(f"Positive feedback received.")
            save_feedback(st.session_state.conversation_id, 1)
            logging.info("Positive feedback saved to database")
    with col2:
        if st.button("-1"):
            st.session_state.count -= 1
            logging.info(
                f"Negative feedback received. New count: {st.session_state.count}"
            )
            save_feedback(st.session_state.conversation_id, -1)
            logging.info("Negative feedback saved to database")

    if st.session_state.count < -1:
        st.write(
            f"It seems you are not satisfied with the answers. Please provide more details or click on the button below to contact a human agent."
        )
        if st.button("Contact human agent"):
            st.write("Redirecting to the contact page...")
            st.write("...")
            st.write("This feature is not yet implemented.")

    # Display recent questions
    st.subheader("Your previous questions")
    relevance_filter = st.selectbox(
        "Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    )
    recent_conversations = get_recent_conversations(
        limit=5, relevance=relevance_filter if relevance_filter != "All" else None
    )
    for conv in recent_conversations:
        st.write(f"Q: {conv['question']}")
        st.write(f"A: {conv['answer']}")
        st.write("---")


logging.info("Streamlit app loop completed")


if __name__ == "__main__":
    logging.info("OLX Assistant application launched")
    main()
