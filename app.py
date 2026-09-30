import streamlit as st
import joblib
import re

from models.text_preprocessor import clean_text


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Spam SMS & Email Classifier",
    page_icon="📩",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/spam_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==================================================
# HEADER
# ==================================================

st.title("📩 Spam SMS & Email Classifier")

st.write(
    "An NLP-based machine learning application that "
    "classifies SMS and email messages as spam or legitimate."
)

st.divider()


# ==================================================
# MESSAGE INPUT
# ==================================================

st.subheader("Enter Your Message")

message = st.text_area(
    "SMS / Email",
    height=170,
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize. Claim now!"
    ),
    label_visibility="collapsed"
)


# ==================================================
# MESSAGE STATISTICS
# ==================================================

if message.strip():

    character_count = len(message)

    word_count = len(message.split())

    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            message
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Characters",
            character_count
        )

    with col2:
        st.metric(
            "Words",
            word_count
        )

    with col3:
        st.metric(
            "URLs",
            url_count
        )


st.write("")


# ==================================================
# BUTTON
# ==================================================

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    check_button = st.button(
        "🔍 Analyze Message",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

if check_button:

    if not message.strip():

        st.warning(
            "Please enter a message before analyzing."
        )

    else:

        # Clean text
        cleaned_message = clean_text(
            message
        )

        # TF-IDF transformation
        message_tfidf = vectorizer.transform(
            [cleaned_message]
        )

        # Prediction probabilities
        probabilities = model.predict_proba(
            message_tfidf
        )[0]

        ham_probability = probabilities[0]

        spam_probability = probabilities[1]


        # Final prediction
        if spam_probability >= 0.5:

            prediction = "SPAM"

        else:

            prediction = "NOT SPAM"


        # ==================================================
        # RESULT
        # ==================================================

        st.divider()

        st.subheader("Prediction Result")


        if prediction == "SPAM":

            st.error(
                "🚨 This message is likely SPAM."
            )

        else:

            st.success(
                "✅ This message is likely NOT SPAM."
            )


        # ==================================================
        # PROBABILITY
        # ==================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Legitimate Probability",
                f"{ham_probability * 100:.2f}%"
            )


        # ==================================================
        # PROBABILITY BAR
        # ==================================================

        st.write("### Spam Probability")

        st.progress(
            spam_probability
        )


        # ==================================================
        # MESSAGE INDICATORS
        # ==================================================

        st.write("### Message Indicators")

        indicator_col1, indicator_col2, indicator_col3 = st.columns(3)


        # URL indicator
        with indicator_col1:

            if url_count > 0:

                st.warning(
                    "🔗 Contains URL"
                )

            else:

                st.success(
                    "🔗 No URL detected"
                )


        # Urgency indicator
        urgency_words = [
            "urgent",
            "now",
            "immediately",
            "claim",
            "limited",
            "winner",
            "congratulations"
        ]

        found_urgency = [
            word
            for word in urgency_words
            if word in message.lower()
        ]

        with indicator_col2:

            if found_urgency:

                st.warning(
                    "⚠️ Urgency-related words detected"
                )

            else:

                st.success(
                    "✓ No major urgency words"
                )


        # Money indicator
        money_pattern = r"[$£€₹]|\b\d+\s?(dollars|rupees|rs|usd|inr)\b"

        money_found = re.search(
            money_pattern,
            message.lower()
        )

        with indicator_col3:

            if money_found:

                st.warning(
                    "💰 Money-related content detected"
                )

            else:

                st.success(
                    "💰 No money indicator detected"
                )


        # ==================================================
        # MODEL INFORMATION
        # ==================================================

        st.divider()

        st.caption(
            "Model: Logistic Regression | "
            "Text Representation: TF-IDF | "
            "Features: Unigrams + Bigrams"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Spam SMS & Email Classifier • "
    "Machine Learning + Natural Language Processing"
)