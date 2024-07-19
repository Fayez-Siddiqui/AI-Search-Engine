from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentActionMessageLog
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import (JsonOutputParser, StrOutputParser)
from typing import List
from typing_extensions import TypedDict
from utils import (
    GraphState,
    web_search,
    transform_query,
    grade_documents,
    generate,
    decide_to_generate,
    grade_generation_v_documents_and_question,
    display_search
)
from langgraph.graph import END, StateGraph, START
import os
from keys import key
import warnings
import streamlit as st
import extra_streamlit_components as stx
import firebase_admin
import jwt
import requests
import streamlit as st
from email_validator import EmailNotValidError, validate_email
from firebase_admin import auth
import math
import time
from contextlib import suppress
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, Final, Optional, Sequence, Union

TITLE="🌕 AI Search Engine"
st.set_page_config(
            page_title=TITLE,
            layout="wide",
            initial_sidebar_state="collapsed",
        )

warnings.filterwarnings("ignore")
user_key=key()
os.environ["GOOGLE_API_KEY"] = user_key.get_key("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = user_key.get_key("TAVILY_API_KEY")
os.environ["LANGSMITH_API_KEY"] = user_key.get_key("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=user_key.get_key("GOOGLE_APPLICATION_CREDENTIALS")
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("display_search",display_search)
# Build graph
workflow.add_edge(START, "web_search")
workflow.add_edge("web_search", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "limit_exceed" : "display_search"
    },
)
workflow.add_edge("display_search",END)


# Compile

ans = workflow.compile()


llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
      convert_system_message_to_human = True,
      verbose = True,
)

prompt =PromptTemplate(
    template='''you are a renowned editor use given article and give a breif summary in 1-2 paragraphs.do not mention the article and act as its your original work. 
Article: {Article}  
Summary:''',
    input_variables=["Article"],
)
chain = prompt | llm | StrOutputParser()


firebaseConfig = user_key.get_firebaseConfig()
#firebase=pyrebase.initialize_app(firebaseConfig)
#auth=firebase.auth()

#db=firebase.database()
#storage=firebase.storage()

POST_REQUEST_URL_BASE: Final = "https://identitytoolkit.googleapis.com/v1/accounts:"
post_request = partial(
    requests.post,
    headers={"content-type": "application/json; charset=UTF-8"},
    timeout=10,
)
success = partial(st.success, icon="✅")
error = partial(st.error, icon="🚨")


def pretty_title(title: str) -> None:
    """Make a centered title, and give it a red line. Adapted from
    'streamlit_extras.colored_headers' package.
    Parameters:
    -----------
    title : str
        The title of your page.
    """
    st.markdown(
        f"<h2 style='text-align: center'>{title}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<hr style="background-color: #ff4b4b; margin-top: 0;'
            ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">'
        ),
        unsafe_allow_html=True,
    )


def parse_error_message(response: requests.Response) -> str:
    """
    Parses an error message from a requests.Response object and makes it look better.
    Parameters:
        response (requests.Response): The response object to parse.
    Returns:
        str: Prettified error message.
    Raises:
        KeyError: If the 'error' key is not present in the response JSON.
    """
    return (
        response.json()["error"]["message"]
        .casefold()
        .replace("_", " ")
        .replace("email", "e-mail")
    )


def authenticate_user(
    email: str, password: str, require_email_verification: bool = True
) -> Optional[Dict[str, Union[str, bool, int]]]:
    """
    Authenticates a user with the given email and password using the Firebase Authentication
    REST API.
    Parameters:
        email (str): The email address of the user to authenticate.
        password (str): The password of the user to authenticate.
        require_email_verification (bool): Specify whether a user has to be e-mail verified to
        be authenticated
    Returns:
        dict or None: A dictionary containing the authenticated user's ID token, refresh token,
        and other information, if authentication was successful. Otherwise, None.
    Raises:
        requests.exceptions.RequestException: If there was an error while authenticating the user.
    """

    url = f"{POST_REQUEST_URL_BASE}signInWithPassword?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
        "emailVerified": require_email_verification,
    }
    response = post_request(url, json=payload)
    if response.status_code != 200:
        error(f"Authentication failed: {parse_error_message(response)}")
        return None
    response = response.json()
    if require_email_verification and "idToken" not in response:
        error("Invalid e-mail or password.")
        return None
    return response


def forgot_password_form(preauthorized: Union[str, Sequence[str], None]) -> None:
    """Creates a Streamlit widget to reset a user's password. Authentication uses
    the Firebase Authentication REST API.
    Parameters:
        preauthorized (Union[str, Sequence[str], None]): An optional domain or a list of
        domains which are authorized to register.
    """

    with st.form("Forgot password"):
        email = st.text_input("E-mail", key="forgot_password")
        if not st.form_submit_button("Reset password"):
            return None
    if "@" not in email and isinstance(preauthorized, str):
        email = f"{email}@{preauthorized}"

    url = f"{POST_REQUEST_URL_BASE}sendOobCode?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {"requestType": "PASSWORD_RESET", "email": email}
    response = post_request(url, json=payload)
    if response.status_code == 200:
        return success(f"Password reset link has been sent to {email}")
    return error(f"Error sending password reset email: {parse_error_message(response)}")


def register_user_form(preauthorized: Union[str, Sequence[str], None]) -> None:
    """Creates a Streamlit widget for user registration.
    Password strength is validated using entropy bits (the power of the password alphabet).
    Upon registration, a validation link is sent to the user's email address.
    Parameters:
        preauthorized (Union[str, Sequence[str], None]): An optional domain or a list of
        domains which are authorized to register.
    """

    with st.form(key="register_form"):
        email, name, password, confirm_password, register_button = (
            st.text_input("E-mail"),
            st.text_input("Name"),
            st.text_input("Password", type="password"),
            st.text_input("Confirm password", type="password"),
            st.form_submit_button(label="Submit"),
        )
    if not register_button:
        return None
    # Below are some checks to ensure proper and secure registration
    if password != confirm_password:
        return error("Passwords do not match")
    if not name:
        return error("Please enter your name")
    if "@" not in email and isinstance(preauthorized, str):
        email = f"{email}@{preauthorized}"
    if preauthorized and not email.endswith(preauthorized):
        return error("Domain not allowed")
    try:
        validate_email(email, check_deliverability=True)
    except EmailNotValidError as e:
        return error(e)

    # Need a password that has minimum 66 entropy bits (the power of its alphabet)
    # I multiply this number by 1.5 to display password strength with st.progress
    # For an explanation, read this -
    # https://en.wikipedia.org/wiki/Password_strength#Entropy_as_a_measure_of_password_strength
    alphabet_chars = len(set(password))
    strength = int(len(password) * math.log2(alphabet_chars) * 1.5)
    if strength < 40:
        st.progress(strength)
        return st.warning(
            "Password is too weak. Please choose a stronger password.", icon="⚠️"
        )
    auth.create_user(
        email=email, password=password, display_name=name, email_verified=False
    )
    # Having registered the user, send them a verification e-mail
    token = authenticate_user(email, password, require_email_verification=False)[
        "idToken"
    ]
    url = f"{POST_REQUEST_URL_BASE}sendOobCode?key={st.secrets['FIREBASE_API_KEY']}"
    payload = {"requestType": "VERIFY_EMAIL", "idToken": token}
    response = post_request(url, json=payload)
    if response.status_code != 200:
        return error(f"Error sending verification email: {parse_error_message(response)}")
    success(
        "Your account has been created successfully. To complete the registration process, "
        "please verify your email address by clicking on the link we have sent to your inbox."
    )
    return st.balloons()


def update_password_form() -> None:
    """Creates a Streamlit widget to update a user's password."""

    # Get the email and password from the user
    new_password = st.text_input("New password", key="new_password")
    # Attempt to log the user in
    if not st.button("Update password"):
        return None
    user = auth.get_user_by_email(st.session_state["username"])
    auth.update_user(user.uid, password=new_password)
    return success("Successfully updated user password.")


def update_display_name_form(
    cookie_manager: stx.CookieManager, cookie_name: str, cookie_expiry_days: int = 30
) -> None:
    """Creates a Streamlit widget to update a user's display name.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    """

    # Get the email and password from the user
    new_name = st.text_input("New name", key="new name")
    # Attempt to log the user in
    if not st.button("Update name"):
        return None
    user = auth.get_user_by_email(st.session_state["username"])
    auth.update_user(user.uid, display_name=new_name)
    st.session_state["name"] = new_name
    # Update the cookie as well
    exp_date = datetime.utcnow() + timedelta(days=cookie_expiry_days)
    cookie_manager.set(
        cookie_name,
        token_encode(exp_date),
        expires_at=exp_date,
    )
    return success("Successfully updated user display name.")


def token_encode(exp_date: datetime) -> str:
    """Encodes a JSON Web Token (JWT) containing user session data for passwordless
    reauthentication.
    Parameters
    ----------
    exp_date : datetime
        The expiration date of the JWT.
    Returns
    -------
    str
        The encoded JWT cookie string for reauthentication.
    Notes
    -----
    The JWT contains the user's name, username, and the expiration date of the JWT in
    timestamp format. The `st.secrets["COOKIE_KEY"]` value is used to sign the JWT with
    the HS256 algorithm.
    """
    return jwt.encode(
        {
            "name": st.session_state["name"],
            "username": st.session_state["username"],
            "exp_date": exp_date.timestamp(),
        },
        st.secrets["COOKIE_KEY"],
        algorithm="HS256",
    )


def cookie_is_valid(cookie_manager: stx.CookieManager, cookie_name: str) -> bool:
    """Check if the reauthentication cookie is valid and, if it is, update the session state.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Returns
    -------
    bool
        True if the cookie is valid and the session state is updated successfully; False otherwise.
    Notes
    -----
    This function checks if the specified reauthentication cookie is present in the cookies stored by
    the cookie manager, and if it is valid. If the cookie is valid, this function updates the session
    state of the Streamlit app and authenticates the user.
    """

    token = cookie_manager.get(cookie_name)
    if token is None:
        return False
    with suppress(Exception):
        token = jwt.decode(token, st.secrets["COOKIE_KEY"], algorithms=["HS256"])
    if (
        token
        and not st.session_state["logout"]
        and token["exp_date"] > datetime.utcnow().timestamp()
        and {"name", "username"}.issubset(set(token))
    ):
        st.session_state["name"] = token["name"]
        st.session_state["username"] = token["username"]
        st.session_state["authentication_status"] = True
        return True
    return False


def login_form(
    cookie_manager: stx.CookieManager,
    cookie_name: str,
    preauthorized: Union[str, Sequence[str], None],
    cookie_expiry_days: int = 30,
) -> None:
    """Creates a login widget using Firebase REST API and a cookie manager.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Notes
    -----
    If the user has already been authenticated, this function does nothing. Otherwise, it displays
    a login form which prompts the user to enter their email and password. If the login credentials
    are valid and the user's email address has been verified, the user is authenticated and a
    reauthentication cookie is created with the specified expiration date.
    """

    if st.session_state["authentication_status"]:
        return None
    with st.form("Login"):
        email = st.text_input("E-mail")
        if "@" not in email and isinstance(preauthorized, str):
            email = f"{email}@{preauthorized}"
        st.session_state["username"] = email
        password = st.text_input("Password", type="password")
        if not st.form_submit_button("Login"):
            return None

    # Authenticate the user with Firebase REST API
    login_response = authenticate_user(email, password)
    if not login_response:
        return None
    try:
        decoded_token = auth.verify_id_token(login_response["idToken"])
        user = auth.get_user(decoded_token["uid"])
        if not user.email_verified:
            return error("Please verify your e-mail address.")
        # At last, authenticate the user
        st.session_state["name"] = user.display_name
        st.session_state["username"] = user.email
        st.session_state["authentication_status"] = True
        exp_date = datetime.utcnow() + timedelta(days=cookie_expiry_days)
        cookie_manager.set(
            cookie_name,
            token_encode(exp_date),
            expires_at=exp_date,
        )
    except Exception as e:
        error(e)
    return None


def login_panel(
    cookie_manager: stx.CookieManager, cookie_name: str, cookie_expiry_days: int = 30
) -> None:
    """Creates a side panel for logged-in users, preventing the login menu from
    appearing.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Notes
    -----
    If the user is logged in, this function displays two tabs for resetting the user's password
    and updating their display name.
    If the user clicks the "Logout" button, the reauthentication cookie and user-related information
    from the session state is deleted, and the user is logged out.
    """

    if st.button("Logout"):
        cookie_manager.delete(cookie_name)
        st.session_state["logout"] = True
        st.session_state["name"] = None
        st.session_state["username"] = None
        st.session_state["authentication_status"] = None
        return None
    st.write(f"Welcome, *{st.session_state['name']}*!")
    user_tab1, user_tab2 = st.tabs(["Reset password", "Update user details"])
    with user_tab1:
        update_password_form()
    with user_tab2:
        update_display_name_form(cookie_manager, cookie_name, cookie_expiry_days)
    return None


def not_logged_in(
    cookie_manager, cookie_name, preauthorized: Union[str, Sequence[str], None] = None
) -> bool:
    """Creates a tab panel for unauthenticated, preventing the user control sidebar and
    the rest of the script from appearing until the user logs in.
    Parameters
    ----------
     - cookie_manager : stx.CookieManager
        A JWT cookie manager instance for Streamlit
    - cookie_name : str
        The name of the reauthentication cookie.
    - cookie_expiry_days: (optional) str
        An integer representing the number of days until the cookie expires
    Returns
    -------
    Authentication status boolean.
    Notes
    -----
    If the user is already authenticated, the login panel function is called to create a side
    panel for logged-in users. If the function call does not update the authentication status
    because the username/password does not exist in the Firebase database, the rest of the script
    does not get executed until the user logs in.
    """

    early_return = True
    # In case of a first run, pre-populate missing session state arguments
    for key in {"name", "authentication_status", "username", "logout"}.difference(
        set(st.session_state)
    ):
        st.session_state[key] = None

    login_tabs = st.empty()
    with login_tabs:
        login_tab1, login_tab2, login_tab3 = st.tabs(
            ["Login", "Register", "Forgot password"]
        )
        with login_tab1:
            login_form(cookie_manager, cookie_name, preauthorized)
        with login_tab2:
            register_user_form(preauthorized)
        with login_tab3:
            forgot_password_form(preauthorized)

    auth_status = st.session_state["authentication_status"]
    if auth_status is False:
        error("Username/password is incorrect")
        return early_return
    if auth_status is None:
        return early_return
    login_tabs.empty()
    # A workaround for a bug in Streamlit -
    # https://playground.streamlit.app/?q=empty-doesnt-work
    # TLDR: element.empty() doesn't actually seem to work with a multi-element container
    # unless you add a sleep after it.
    time.sleep(0.01)
    return not early_return


def app() -> None:
    """This is a part of a Streamlit app which is only visible if the user is logged in."""
    st.write("Welcome "+st.session_state["name"]+"😎")
    if "messages" not in st.session_state.keys():
                st.session_state.messages = [{"role": "😇", "content": "How may I help you?"}]
                
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "😎", "content": prompt})
        with st.chat_message("😎"):
            st.write(prompt)
    if st.session_state.messages[-1]["role"] != "😇":
        with st.chat_message("😇"):
            with st.spinner("Thinking..."):
                inputs = {"question": prompt}
                for output in ans.stream(inputs):
                    for key, value in output.items():
                        # Node
                        print(f"Node '{key}':")
                        print("\n---\n")
                print(value['generation'])
                documents=value['documents']
                response=""
                temp=""
                for d in documents :
                    print(d.page_content)
                    temp=temp + d.page_content
                temp=chain.invoke({"Article" : temp})
                print(temp)
                response=temp
                response=response + f"\n"
                response = response + value["generation"]
                if len(response)>0:
                    st.write(response)
                else :
                    st.write("Didn't catch you there 😓") 
        message = {"role": "😇", "content": response}
        st.session_state.messages.append(message)
TITLE="🌕 AI Search Engine"

def main() :
        # Hides 'Made with Streamlit'
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # noinspection PyProtectedMember
    if not firebase_admin._apps:
        cred = firebase_admin.credentials.Certificate(
            st.secrets["firebase_auth_token"]
        )
        firebase_admin.initialize_app(cred,{    
        'databaseURL': user_key.get_key('databaseURL')
})
    pretty_title(TITLE)
    cookie_manager, cookie_name = stx.CookieManager(), "login_cookie"

    if not cookie_is_valid(cookie_manager, cookie_name) and not_logged_in(
        cookie_manager, cookie_name, preauthorized="gmail.com"
    ):
        return None

    with st.sidebar:
        login_panel(cookie_manager, cookie_name)

    return app()

if __name__ == "__main__":
    main()
