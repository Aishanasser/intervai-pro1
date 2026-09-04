import json
import os
import streamlit as st
import streamlit.components.v1 as components
import time
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
from pypdf import PdfReader
from ai_engine import (extract_skills, normalize_text, detect_language,
                       ANSWER_WEAK_THRESHOLD, ANSWER_STRONG_THRESHOLD)
from graph_engine import run_cv_jd_pipeline, run_answer_cycle

st.set_page_config(
    page_title="IntervAI Pro | AI Interview Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if st.session_state.dark_mode:
    _bg_base = "#030014"
    _bg_gradient = "radial-gradient(circle at 50% -20%, #1e1145, #030014 60%)"
    _text_color = "#F3F4F6"
    _card_bg = "rgba(255, 255, 255, 0.02)"
    _card_border = "rgba(255, 255, 255, 0.08)"
    _card_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
    _sidebar_bg = "rgba(3, 0, 20, 0.9)"
    _sidebar_border = "rgba(255, 255, 255, 0.05)"
    _gradient_text = "linear-gradient(135deg, #FFFFFF 30%, #A78BFA 100%)"
    _badge_bg = "rgba(124, 58, 237, 0.15)"
    _badge_color = "#C084FC"
    _muted = "#9CA3AF"
    _muted_soft = "#6B7280"
    _hr = "rgba(255, 255, 255, 0.06)"
    _c_violet = "#A78BFA"
    _c_blue = "#60A5FA"
    _c_green = "#34D399"
    _c_cyan = "#06B6D4"
    _c_amber = "#F59E0B"
    _c_yellow = "#FBBF24"
    _c_red = "#F87171"
    _input_bg = "rgba(255, 255, 255, 0.04)"
    _input_border = "rgba(255, 255, 255, 0.14)"
else:
    _bg_base = "#F7F5FC"
    _bg_gradient = "radial-gradient(circle at 50% -20%, #EDE9FE, #F7F5FC 60%)"
    _text_color = "#1F2937"
    _card_bg = "rgba(17, 24, 39, 0.03)"
    _card_border = "rgba(17, 24, 39, 0.08)"
    _card_shadow = "0 8px 32px 0 rgba(109, 40, 217, 0.08)"
    _sidebar_bg = "rgba(255, 255, 255, 0.85)"
    _sidebar_border = "rgba(17, 24, 39, 0.06)"
    _gradient_text = "linear-gradient(135deg, #1F2937 30%, #7C3AED 100%)"
    _badge_bg = "rgba(124, 58, 237, 0.08)"
    _badge_color = "#6D28D9"
    # The light branch darkens every accent. The dark-theme tints (#A78BFA,
    # #34D399 ...) sit around 2:1 contrast on a near-white page, which is
    # below the readable floor; these variants clear 4.5:1.
    _muted = "#4B5563"
    _muted_soft = "#6B7280"
    _hr = "rgba(17, 24, 39, 0.10)"
    _c_violet = "#6D28D9"
    _c_blue = "#1D4ED8"
    _c_green = "#047857"
    _c_cyan = "#0E7490"
    _c_amber = "#B45309"
    _c_yellow = "#B45309"
    _c_red = "#DC2626"
    _input_bg = "#FFFFFF"
    _input_border = "rgba(17, 24, 39, 0.16)"

# Injected custom CSS - includes a permanent fix for the sidebar toggle button disappearing
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: {_bg_base} !important;
        color: {_text_color};
    }}

    .stApp {{
        background: {_bg_gradient};
    }}

    /* === Legibility on the light theme ===
       .streamlit/config.toml pins Streamlit's base theme to "dark", which is
       what gives the app its look out of the box — but it also means every
       element Streamlit renders itself (headings, labels, tab titles,
       expander headers, widget text) is painted near-white by Streamlit's own
       stylesheet. Setting `color` on html/body cannot reach them: those rules
       target the elements directly and win on specificity. So when Dark Mode
       is switched off the background flips to light and that text stays
       white, i.e. invisible. These selectors re-colour Streamlit's elements
       from the same variables that drive the rest of the palette. */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp small, .stApp summary,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-baseweb="tab"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {_text_color};
    }}

    /* Buttons keep white text: they sit on the violet gradient in both
       themes, so the rule above must not reach the label inside them. */
    .stApp .stButton > button, .stApp .stButton > button *,
    .stApp .stDownloadButton > button, .stApp .stDownloadButton > button *,
    .stApp .stFormSubmitButton > button, .stApp .stFormSubmitButton > button * {{
        color: #FFFFFF !important;
    }}

    /* Captions stay deliberately quieter than body text. */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {{
        color: {_muted} !important;
    }}

    /* Inputs carry their own surface so the answer box, the CV uploader and
       the dropdowns stay legible on either background. */
    .stApp textarea, .stApp input,
    .stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"],
    .stApp [data-baseweb="base-input"], .stApp [data-baseweb="base-textarea"],
    .stApp [data-baseweb="select"] > div,
    .stApp [data-testid="stFileUploaderDropzone"] {{
        background-color: {_input_bg} !important;
        color: {_text_color} !important;
        border-color: {_input_border} !important;
    }}

    /* The password reveal icon is a button *inside* the input, so it inherits
       nothing from the rule above. */
    .stApp [data-baseweb="base-input"] button {{
        color: {_text_color} !important;
        background: transparent !important;
    }}

    .stApp textarea::placeholder, .stApp input::placeholder {{
        color: {_muted} !important;
        opacity: 1;
    }}

    .stApp hr {{
        border-color: {_hr};
    }}

    /* Streamlit's secondary buttons — the file uploader's browse control among
       them — keep the dark base surface, so give them one that matches the
       active theme rather than leaving dark text on a dark button. */
    .stApp button[data-testid="stBaseButton-secondary"] {{
        background-color: {_input_bg} !important;
        border-color: {_input_border} !important;
    }}
    .stApp button[data-testid="stBaseButton-secondary"] * {{
        color: {_text_color} !important;
    }}

    /* The uploader's hint line ("5MB per file - PDF") is a bare span that
       Streamlit paints translucent white. */
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] span,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: {_muted} !important;
    }}

    /* Streamlit's default link blue drops under 3:1 on the light background. */
    .stApp a, [data-testid="stSidebar"] a {{
        color: {_c_blue};
    }}

    /* The sidebar collapse/expand arrows. Streamlit renamed this control's
       test id (it is no longer stSidebarCollapseButton), so the gradient rule
       further down no longer reaches it and the icon stayed near-white —
       invisible on the light background. */
    .stApp button[data-testid="stBaseButton-headerNoPadding"] *,
    .stApp button[data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] * {{
        color: {_text_color} !important;
    }}

    /* === 🛠️ Definitive fix for the sidebar toggle button === */
    button[data-testid="stSidebarCollapseButton"] {{
        background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        opacity: 1 !important;
        visibility: visible !important;
        z-index: 999999 !important;
        box-shadow: 0 0 15px rgba(109, 40, 217, 0.6) !important;
        position: fixed !important;
        top: 15px !important;
    }}

    button[data-testid="stSidebarCollapseButton"]:hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 0 25px rgba(109, 40, 217, 0.9) !important;
    }}

    header {{
        background: transparent !important;
    }}

    /* Premium glass-morphism effects for cards */
    .premium-card {{
        background: {_card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {_card_border};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: {_card_shadow};
        transition: all 0.3s ease;
    }}

    .premium-card:hover {{
        border-color: rgba(124, 58, 237, 0.4);
        box-shadow: 0 8px 40px 0 rgba(124, 58, 237, 0.1);
    }}

    .gradient-text {{
        background: {_gradient_text};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -1px;
    }}

    .stButton>button {{
        background: linear-gradient(90deg, #6D28D9 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        width: 100%;
    }}

    [data-testid="stSidebar"] {{
        background-color: {_sidebar_bg} !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid {_sidebar_border};
    }}

    /* Arabic content renders right-to-left. Scoped to generated content
       rather than the whole page: the interface chrome stays English, and
       flipping it would misalign every label. Latin technology names inside
       an Arabic sentence are handled by the browser's bidi algorithm — they
       stay left-to-right on their own, which is why the prompts keep them in
       Latin script. */
    .rtl {{
        direction: rtl;
        text-align: right;
        unicode-bidi: isolate;
    }}

    .badge {{
        background: {_badge_bg};
        color: {_badge_color} !important;
        border: 1px solid rgba(124, 58, 237, 0.3);
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.85rem;
        display: inline-block;
    }}
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* ---- keyframes ---- */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes softGlow {
        0%, 100% { box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35); }
        50%      { box-shadow: 0 6px 34px rgba(99, 102, 241, 0.75); }
    }
    @keyframes badgePulse {
        0%, 100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.45); }
        50%      { transform: scale(1.08); box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
    }
    @keyframes shimmer {
        0%   { background-position: -300px 0; }
        100% { background-position: 300px 0; }
    }

    /* ---- entrance: every block Streamlit renders fades/slides in ----
       The modern data-testid is listed first; the legacy .main .block-container
       is kept only as a fallback for older Streamlit builds (it matches
       nothing on current versions). */
    [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] > div,
    [data-testid="stAppViewContainer"] .main .block-container > div {
        animation: fadeSlideUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    [data-testid="stSidebar"] {
        animation: fadeIn 0.6s ease both;
    }

    /* ---- premium-card: lift + cursor-tracked spotlight ---- */
    .premium-card {
        animation: fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        position: relative;
        overflow: hidden;
    }
    /* the spotlight follows the cursor via --mx/--my set in the JS below */
    .premium-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: radial-gradient(
            520px circle at var(--mx, 50%) var(--my, 50%),
            rgba(124, 58, 237, 0.22),
            transparent 45%
        );
        opacity: 0;
        transition: opacity 0.35s ease;
        pointer-events: none;
        z-index: 0;
    }
    .premium-card:hover::before { opacity: 1; }
    .premium-card > * { position: relative; z-index: 1; }
    .premium-card:hover {
        transform: translateY(-6px) scale(1.008);
        border-color: rgba(124, 58, 237, 0.55) !important;
        box-shadow: 0 14px 44px rgba(124, 58, 237, 0.22) !important;
    }

    /* ---- gradient headline text: living shimmer (wider travel, faster) ---- */
    .gradient-text {
        background-size: 300% 300%;
        animation: gradientShift 4s ease infinite;
    }

    /* ---- badge: breathing pulse with an expanding halo ---- */
    .badge {
        animation: badgePulse 2.2s ease-in-out infinite;
    }

    /* ---- buttons: lift, glow, and a light sheen sweeping across on hover ---- */
    .stButton > button {
        position: relative;
        overflow: hidden;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease !important;
    }
    .stButton > button::after {
        content: "";
        position: absolute;
        top: 0;
        left: -60%;
        width: 40%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.38),
            transparent
        );
        transform: skewX(-20deg);
        transition: left 0.55s ease;
        pointer-events: none;
    }
    .stButton > button:hover::after { left: 130%; }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        animation: softGlow 1.5s ease-in-out infinite;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ---- cursor-tracked ambient glow (element injected by the JS below) ---- */
    #cursor-glow {
        position: fixed;
        top: 0;
        left: 0;
        width: 460px;
        height: 460px;
        margin-left: -230px;
        margin-top: -230px;
        border-radius: 50%;
        background: radial-gradient(
            circle,
            rgba(124, 58, 237, 0.20) 0%,
            rgba(79, 70, 229, 0.10) 35%,
            transparent 70%
        );
        pointer-events: none;
        z-index: 0;
        opacity: 0;
        transition: opacity 0.4s ease;
        will-change: transform;
    }
    #cursor-glow.visible { opacity: 1; }

    /* ---- sidebar nav radio options: smooth hover shift ---- */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        transition: transform 0.2s ease, opacity 0.2s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        transform: translateX(-4px);
        opacity: 0.85;
    }

    /* ---- expanders: gentle open/close feel ---- */
    [data-testid="stExpander"] {
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.08);
    }

    /* ---- text inputs / text areas: focus glow instead of a hard snap ---- */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        transition: box-shadow 0.25s ease, border-color 0.25s ease !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25) !important;
    }

    /* ---- progress bar fill: subtle shimmer while it moves ---- */
    [data-testid="stProgress"] > div > div > div {
        background-image: linear-gradient(
            90deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.35) 50%,
            rgba(255,255,255,0) 100%
        ), linear-gradient(90deg, #6D28D9, #4F46E5) !important;
        background-size: 200px 100%, 100% 100%;
        animation: shimmer 1.4s linear infinite;
    }

    /* ---- metric-style small cards (m1..m4 in dashboard) get a tiny hover lift too ---- */
    .premium-card h2 {
        transition: transform 0.2s ease;
    }
    .premium-card:hover h2 {
        transform: scale(1.03);
    }

    /* ---- respect users who prefer reduced motion ---- */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


components.html("""
<script>
(function () {
    const doc = window.parent.document;
    if (doc.getElementById("cursor-glow")) { return; }   // already installed

    // Ambient glow that trails the cursor across the whole page.
    const glow = doc.createElement("div");
    glow.id = "cursor-glow";
    doc.body.appendChild(glow);

    let targetX = 0, targetY = 0, currentX = 0, currentY = 0, running = false;

    function animate() {
        // Ease toward the cursor so the glow trails rather than snaps.
        currentX += (targetX - currentX) * 0.12;
        currentY += (targetY - currentY) * 0.12;
        glow.style.transform = `translate(${currentX}px, ${currentY}px)`;
        if (Math.abs(targetX - currentX) > 0.5 || Math.abs(targetY - currentY) > 0.5) {
            requestAnimationFrame(animate);
        } else {
            running = false;
        }
    }

    doc.addEventListener("mousemove", function (e) {
        targetX = e.clientX;
        targetY = e.clientY;
        glow.classList.add("visible");
        if (!running) { running = true; requestAnimationFrame(animate); }

        // Per-card spotlight: feed the cursor position to the hovered card.
        const card = e.target.closest ? e.target.closest(".premium-card") : null;
        if (card) {
            const r = card.getBoundingClientRect();
            card.style.setProperty("--mx", (e.clientX - r.left) + "px");
            card.style.setProperty("--my", (e.clientY - r.top) + "px");
        }
    }, { passive: true });

    doc.addEventListener("mouseleave", function () {
        glow.classList.remove("visible");
    });
})();
</script>
""", height=0)

# Connection details come from the environment, not from the source.
#
# They used to be hardcoded to localhost, which means "the machine running this
# code". That works while the app and MySQL sit on the same laptop, and fails
# silently everywhere else: a participant running the app on their own computer
# resolves localhost to their machine, finds no database, and every save call
# quietly does nothing. For a study whose whole output is the saved responses,
# that is the worst possible failure mode.
#
# Defaults keep the local XAMPP setup working unchanged, so nothing breaks
# while a shared database is being set up.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "intervai_db"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Managed providers (Aiven among them) refuse unencrypted connections, while a
# local XAMPP install has no certificate at all. Pointing DB_SSL_CA at the
# provider's CA file switches TLS on; leaving it unset keeps the local setup
# working exactly as before.
_DB_SSL_CA = os.getenv("DB_SSL_CA", "").strip()
if _DB_SSL_CA:
    DB_CONFIG["ssl_ca"] = _DB_SSL_CA
    DB_CONFIG["ssl_verify_cert"] = True

# Remembered so the failure is reported once per session rather than on every
# rerun, which in Streamlit would mean on every keystroke.
_DB_ERROR = None


def get_db_connection():
    global _DB_ERROR
    try:
        conn = mysql.connector.connect(connection_timeout=8, **DB_CONFIG)
        _DB_ERROR = None
        return conn
    except Error as e:
        _DB_ERROR = str(e)
        return None


@st.cache_data(ttl=30, show_spinner=False)
def db_status() -> tuple[bool, str]:
    """Whether the database is reachable, and why not if it isn't.

    Cached for 30 seconds because Streamlit re-runs the whole script on every
    interaction — without this, the status badge would open a fresh connection
    to a remote server on each keystroke, and every one of those is a round
    trip to another country.

    The TTL is short on purpose: a database that drops mid-interview must show
    as down within seconds, not stay green because the answer was cached.
    Actual saves are never cached — they always attempt a real connection.
    """
    conn = get_db_connection()
    if conn:
        conn.close()
        return True, f"{DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return False, _DB_ERROR or "unknown error"

def init_db():
    """Create the schema if it is missing.

    Wrapped because this runs at import time: an unhandled exception here
    stops the whole app from loading, and a silent skip leaves every later
    query failing against tables that were never created.
    """
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                job_title VARCHAR(255) NOT NULL,
                experience_level VARCHAR(100),
                job_description TEXT,
                score INT DEFAULT NULL,
                ai_evaluation TEXT,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_qa (
                id INT AUTO_INCREMENT PRIMARY KEY,
                interview_id INT,
                question_text TEXT NOT NULL,
                user_answer TEXT,
                score FLOAT NULL,
                evaluation_json TEXT NULL,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE
            )
        ''')

        # Migration for databases created before per-answer scoring existed.
        # CREATE TABLE IF NOT EXISTS leaves an older table untouched, so the
        # new columns have to be added explicitly; MySQL has no
        # ADD COLUMN IF NOT EXISTS, hence the check against information_schema.
        cursor.execute('''
            SELECT COLUMN_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'interview_qa'
        ''', (DB_CONFIG["database"],))
        existing = {r[0] for r in cursor.fetchall()}
        for column, ddl in (("score", "ADD COLUMN score FLOAT NULL"),
                            ("evaluation_json", "ADD COLUMN evaluation_json TEXT NULL")):
            if column not in existing:
                cursor.execute(f"ALTER TABLE interview_qa {ddl}")

        # interviews.score was created as DEFAULT 0, which makes an interview
        # that was never finished indistinguishable from one genuinely scored
        # zero. The dashboard filters on `score is not None`, so a walked-away
        # session was being counted as a real 0% — dragging the candidate's
        # average down and showing "declining" on the progress panel. NULL is
        # the only value that means "no score yet".
        #
        # SET DEFAULT NULL, not DROP DEFAULT. Dropping the default removes the
        # implicit DEFAULT NULL along with the 0, and Aiven runs with
        # STRICT_ALL_TABLES: an INSERT that omits a column with no default at
        # all then fails outright with "Field 'score' doesn't have a default
        # value" — even though the column is nullable. That is exactly what
        # create_interview_session() does, so the earlier DROP DEFAULT stopped
        # the platform from starting any new interview at all. This statement
        # is idempotent, so it also repairs a database left in that state.
        cursor.execute("ALTER TABLE interviews ALTER COLUMN score SET DEFAULT NULL")

        # Resume support. Everything an in-progress interview needs lives in
        # st.session_state, which is per browser session and is wiped by a
        # logout, a refresh, or the host putting the app to sleep. The answers
        # were already durable — each one is written as it is submitted — but
        # the question queue and the agent's budget counters were not, so a
        # candidate who dropped out could only start over. This column holds
        # them as JSON, and is cleared when the interview is finalised.
        cursor.execute("""
            SELECT COLUMN_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'interviews'
        """, (DB_CONFIG["database"],))
        if "resume_json" not in {r[0] for r in cursor.fetchall()}:
            cursor.execute("ALTER TABLE interviews ADD COLUMN resume_json MEDIUMTEXT NULL")

        conn.commit()
        return True
    except Error as e:
        # Surfaced rather than swallowed: without the schema nothing works, and
        # the cause (permissions, wrong database name) belongs on screen.
        st.error(f"⚠️ Could not create the database schema: {e}")
        return False
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

# Auto-create/repair tables on startup
init_db()

def create_user(name, email, password):
    """Returns (ok, reason). `reason` distinguishes a duplicate email from a
    database failure — returning a bare False for both told a user whose
    database was down that their email was already taken, which is a
    misleading message and sends them to fix the wrong thing."""
    conn = get_db_connection()
    if not conn:
        return False, "unavailable"
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                       (name, email, password))
        conn.commit()
        return True, "created"
    except mysql.connector.IntegrityError:
        return False, "duplicate"
    except Error:
        return False, "unavailable"
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

def authenticate_user(email, password):
    """Returns (user_id, name) on success, None on wrong credentials, and the
    string "unavailable" when the database could not be reached.

    The three cases used to collapse into a bare None, so a user whose database
    was down was told their email or password was wrong — sending them to
    change a password that was never the problem.
    """
    conn = get_db_connection()
    if not conn:
        return "unavailable"
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM users WHERE email=%s AND password=%s',
                       (email, password))
        result = cursor.fetchone()
        return (result[0], result[1]) if result else None
    except Error:
        return "unavailable"
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

def create_interview_session(user_id, job_title, exp_level, job_desc):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # score stays NULL until the interview is finished: see the
            # migration note in init_db().
            cursor.execute('INSERT INTO interviews (user_id, job_title, experience_level, job_description, score) VALUES (%s, %s, %s, %s, NULL)', (user_id, job_title, exp_level, job_desc))
            conn.commit()
            return cursor.lastrowid
        except Error:
            return None
        finally:
            cursor.close()
            conn.close()
    return None

def save_question_answer(interview_id, question_text, user_answer,
                         score=None, evaluation=None):
    """Persist one answer together with its evaluation.

    Returns True on success. The caller is expected to surface a failure:
    this used to return nothing and skip silently when the database was
    unreachable, so an entire interview could be conducted and lost without
    anyone noticing.
    """
    conn = get_db_connection()
    if not conn:
        st.error(f"⚠️ Could not save this answer — database unreachable "
                 f"({_DB_ERROR}). Your answers are NOT being recorded.")
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO interview_qa '
            '(interview_id, question_text, user_answer, score, evaluation_json) '
            'VALUES (%s, %s, %s, %s, %s)',
            (interview_id, question_text, user_answer, score,
             json.dumps(evaluation, ensure_ascii=False) if evaluation else None))
        conn.commit()
        return True
    except Error as e:
        st.error(f"⚠️ Could not save this answer: {e}")
        return False
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

# Fields that together describe "where this interview had got to". Answers
# are NOT among them — those are already rows in interview_qa — but the
# evaluations are, because the final report averages them and re-reading them
# out of the answer rows would mean re-parsing JSON that is already in memory.
_RESUME_FIELDS = (
    "interview_queue", "current_question", "interview_answers",
    "interview_evaluations", "covered_skills", "probe_depth", "off_plan_used",
    "generated_questions", "skill_gap", "jd_skills", "cv_skills",
    "interview_language", "mock_questions",
)


def save_interview_progress(interview_id) -> None:
    """Snapshot the live interview so it can be picked up later.

    Best effort by design: this runs after every answer, and a failure here
    must not cost the candidate the answer they just submitted — that is
    already stored by save_question_answer(). So it stays silent, and the
    worst case is the interview being non-resumable, which is where it was
    before this existed.
    """
    if not interview_id:
        return
    conn = get_db_connection()
    if not conn:
        return
    try:
        payload = json.dumps({f: st.session_state.get(f) for f in _RESUME_FIELDS},
                             ensure_ascii=False, default=str)
        cursor = conn.cursor()
        cursor.execute("UPDATE interviews SET resume_json = %s WHERE id = %s",
                       (payload, interview_id))
        conn.commit()
    except (Error, TypeError, ValueError):
        pass
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass


def load_unfinished_interview(user_id):
    """Newest interview for this user that was never finished and has a
    snapshot. Returns (row, state) or (None, None)."""
    conn = get_db_connection()
    if not conn:
        return None, None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, job_title, session_date, resume_json FROM interviews "
            "WHERE user_id = %s AND score IS NULL AND resume_json IS NOT NULL "
            "ORDER BY id DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if not row:
            return None, None
        return row, json.loads(row["resume_json"])
    except (Error, ValueError, TypeError):
        return None, None
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass


def finalize_interview(interview_id, evaluation_text):
    """Average the stored answer rows into the interview's final score.

    Returns the score written (0-100), or None if nothing was written.

    This had neither an error branch nor a try block: if the database was
    unreachable the update was skipped in silence, so the candidate saw a score
    on screen while the stored row kept its default of 0. For a study whose
    only output is the stored rows, a score that exists on screen and nowhere
    else is worse than no score at all.

    The score is now computed HERE, from interview_qa, instead of being passed
    in from st.session_state. The caller used to average a list held in browser
    session state and fall back to `0` when that list was empty — so a session
    whose state had been cleared (a refresh, the host sleeping the app) wrote a
    confident 0 over an interview whose answers were all sitting in the
    database, correctly scored. One real candidate was recorded at 0% with a
    true average of 58%. The answer rows are the durable record; averaging them
    cannot disagree with them, and when there is nothing to average the column
    is left NULL, which is the only value that means "not scored".
    """
    conn = get_db_connection()
    if not conn:
        st.error("⚠️ Could not save your final score — database unreachable "
                 f"({_DB_ERROR}).")
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT AVG(score) FROM interview_qa '
            'WHERE interview_id = %s AND score IS NOT NULL', (interview_id,))
        row = cursor.fetchone()
        if not row or row[0] is None:
            st.warning("⚠️ No scored answers were found for this interview, so "
                       "no final score was recorded.")
            return None
        score = round(100 * float(row[0]))
        # resume_json is cleared here: a finished interview must not show up
        # as resumable on the next login.
        cursor.execute(
            'UPDATE interviews SET score = %s, ai_evaluation = %s, '
            'resume_json = NULL WHERE id = %s',
            (score, evaluation_text, interview_id))
        conn.commit()
        if cursor.rowcount == 0:
            # The update ran but matched nothing — a wrong or deleted id.
            st.warning(f"⚠️ No interview row #{interview_id} was updated.")
            return None
        return score
    except Error as e:
        st.error(f"⚠️ Could not save your final score: {e}")
        return None
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

def get_skill_progress(user_id):
    """Per-skill scores across every session this user has completed.

    Reads the answer rows rather than the interview totals, because "you
    improved" is only useful when it names the skill: an average that moves
    from 60 to 68 tells a candidate nothing about what to practise next, while
    "Docker 45% -> 82%, Kubernetes still 30%" tells them exactly.

    Returns {skill: [(session_date, mean_score), ...]} ordered oldest first.
    """
    conn = get_db_connection()
    if not conn:
        return {}, False
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT i.id, i.session_date, a.question_text, a.score,
                   a.evaluation_json
            FROM interviews i
            JOIN interview_qa a ON a.interview_id = i.id
            WHERE i.user_id = %s AND a.score IS NOT NULL
            ORDER BY i.session_date, a.id
        """, (user_id,))
        rows = cursor.fetchall()
    except Error:
        return {}, False
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

    # The skill each answer belonged to lives in the evaluation payload; older
    # rows predate that field, so they are grouped by interview instead of
    # being dropped.
    per_session = {}
    for r in rows:
        skill = "general"
        if r.get("evaluation_json"):
            try:
                skill = json.loads(r["evaluation_json"]).get("targets_skill") or "general"
            except (ValueError, TypeError):
                pass
        per_session.setdefault((r["id"], r["session_date"]), {}).setdefault(skill, []).append(r["score"])

    progress = {}
    for (_, when), skills in sorted(per_session.items(), key=lambda kv: kv[0][1]):
        for skill, scores in skills.items():
            progress.setdefault(skill, []).append((when, sum(scores) / len(scores)))
    return progress, True


def get_weakest_answers(user_id, limit=3):
    """The weakest answers from this user's most recent interview.

    Read from the database rather than from session state. The dashboard used
    st.session_state.interview_answers, which is populated only while an
    interview is running in the current browser session — so anyone who
    finished an interview and logged back in later was told to "complete an
    interview to see recommendations", with a full set of stored evaluations
    sitting in the database the whole time.

    Scoped to the latest interview because advice should describe where the
    candidate stands now, not repeat a weakness from months ago they may
    already have fixed.
    """
    conn = get_db_connection()
    if not conn:
        return [], False
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT a.score, a.evaluation_json
            FROM interview_qa a
            WHERE a.interview_id = (
                SELECT i.id FROM interviews i
                JOIN interview_qa qa ON qa.interview_id = i.id
                WHERE i.user_id = %s AND qa.score IS NOT NULL
                ORDER BY i.session_date DESC, i.id DESC LIMIT 1
            ) AND a.score IS NOT NULL
            ORDER BY a.score ASC LIMIT %s
        """, (user_id, limit))
        rows = cursor.fetchall()
    except Error:
        return [], False
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass

    out = []
    for r in rows:
        ev = {}
        if r.get("evaluation_json"):
            try:
                ev = json.loads(r["evaluation_json"])
            except (ValueError, TypeError):
                ev = {}
        if ev.get("feedback"):
            out.append({"score": r["score"],
                        "targets_skill": ev.get("targets_skill") or "General",
                        "feedback": ev["feedback"]})
    return out, True


def get_user_interview_history(user_id):
    """Returns (rows, ok). `ok` is False when the database could not be read.

    An empty list used to mean both "this user has no interviews" and "the
    database is down". The dashboard then displayed a confident zero for
    someone who might have ten sessions stored — a wrong number is worse than
    a visible error.
    """
    conn = get_db_connection()
    if not conn:
        return [], False
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT id, job_title, experience_level, score, ai_evaluation, '
            'session_date FROM interviews WHERE user_id=%s ORDER BY session_date ASC',
            (user_id,))
        return cursor.fetchall(), True
    except Error:
        return [], False
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass


if "page" not in st.session_state: st.session_state.page = "Landing"
if "is_logged_in" not in st.session_state: st.session_state.is_logged_in = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "cv_uploaded" not in st.session_state: st.session_state.cv_uploaded = False
if "cv_skills" not in st.session_state: st.session_state.cv_skills = {}
if "skill_gap" not in st.session_state: st.session_state.skill_gap = {}
if "generated_questions" not in st.session_state: st.session_state.generated_questions = []
if "interview_answers" not in st.session_state: st.session_state.interview_answers = []
# The live interview queue. It starts as the generated plan but the agent may
# insert follow-ups or drop questions mid-interview, so it is kept separately
# from `generated_questions` (which stays as the record of what was planned).
if "interview_queue" not in st.session_state: st.session_state.interview_queue = []
if "interview_evaluations" not in st.session_state: st.session_state.interview_evaluations = []
if "final_score" not in st.session_state: st.session_state.final_score = None
if "interview_language" not in st.session_state: st.session_state.interview_language = "en"
if "covered_skills" not in st.session_state: st.session_state.covered_skills = []
if "probe_depth" not in st.session_state: st.session_state.probe_depth = 0
if "off_plan_used" not in st.session_state: st.session_state.off_plan_used = 0
if "cv_file_name" not in st.session_state: st.session_state.cv_file_name = None
if "interview_started" not in st.session_state: st.session_state.interview_started = False
if "current_question" not in st.session_state: st.session_state.current_question = 0
if "current_interview_id" not in st.session_state: st.session_state.current_interview_id = None
if "mock_questions" not in st.session_state:
    st.session_state.mock_questions = [
        "Tell me about a challenging technical project you worked on, specifically focusing on hardware-software integration.",
        "How do you approach debugging a memory allocation or latency issue in an embedded or real-time system?",
        "Why do you want to join our company, and how do your skills align with the Job Description provided?"
    ]

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_landing():
    st.markdown("<div style='text-align: center; padding: 60px 0 40px 0;'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>✨ Next-Gen AI Mock Interviews</span>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 4rem; margin-top: 15px;' class='gradient-text'>Ace your next job interview <br>with the power of AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.25rem; color: {_muted}; max-width: 700px; margin: 20px auto;'>Upload your CV, enter the job description, and let the AI interview engine test you and give you a detailed evaluation that guarantees you're ready.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Start training for free now", key="hero_cta"):
            navigate_to("Auth")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color: {_hr}; margin: 40px 0;'>", unsafe_allow_html=True)

    # Features
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;' class='gradient-text'>Platform Features</h2>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(f'<div class="premium-card"><h3 style="color: {_c_violet};">📄 CV Analysis</h3><p style="color: {_muted};">An LLM-powered processing system that analyzes the strengths, weaknesses, and experience in your CV.</p></div>', unsafe_allow_html=True)
    with f_col2:
        st.markdown(f'<div class="premium-card"><h3 style="color: {_c_blue};">🎯 Smart Live Simulation</h3><p style="color: {_muted};">Dynamically generates technical and behavioral questions tailored to the company and job description.</p></div>', unsafe_allow_html=True)
    with f_col3:
        st.markdown(f'<div class="premium-card"><h3 style="color: {_c_green};">📊 Scoring & Evaluation</h3><p style="color: {_muted};">Get a complete dashboard showing your strengths and precise mistakes, along with professional alternative phrasing.</p></div>', unsafe_allow_html=True)

    # Pricing
    st.markdown(f"<hr style='border-color: {_hr}; margin: 40px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;' class='gradient-text'>Flexible Subscription Plans</h2>", unsafe_allow_html=True)

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown(f'<div class="premium-card" style="border-top: 3px solid {_muted};"><h3>Free Plan</h3><h2 style="margin: 15px 0;">$0 <span style="font-size: 1rem; color: {_muted};">/ month</span></h2><p style="color: {_muted};">• One full interview per month<br>• Basic CV analysis<br>• Simplified report format</p></div>', unsafe_allow_html=True)
        if st.button("Choose the Free Plan", key="btn_free"): navigate_to("Auth")

    with p_col2:
        st.markdown(f'<div class="premium-card" style="border-top: 3px solid #7C3AED; background: rgba(124, 58, 237, 0.03);"><h3>Pro Plan 🔥</h3><h2 style="margin: 15px 0;">$29 <span style="font-size: 1rem; color: {_muted};">/ month</span></h2><p style="color: {_c_violet};">• Unlimited AI-powered interviews<br>• Advanced analysis & JD matching<br>• Precise AI answer evaluation</p></div>', unsafe_allow_html=True)
        if st.button("Subscribe to Pro Now", key="btn_pro"): navigate_to("Auth")

    with p_col3:
        st.markdown(f'<div class="premium-card" style="border-top: 3px solid {_c_cyan};"><h3>Premium Plan 👑</h3><h2 style="margin: 15px 0;">$79 <span style="font-size: 1rem; color: {_muted};">/ month</span></h2><p style="color: {_muted};">• All Pro plan features<br>• Live voice interview simulation<br>• Custom reports, shareable with companies</p></div>', unsafe_allow_html=True)
        if st.button("Subscribe to Premium Now", key="btn_premium"): navigate_to("Auth")

# --- PAGE: AUTHENTICATION (CONNECTED TO MYSQL) ---
def render_auth():
    st.markdown("<div style='max-width: 450px; margin: 40px auto;'>", unsafe_allow_html=True)
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;' class='gradient-text'>Welcome to IntervAI Pro</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Create New Account"])
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log In", key="btn_login_submit"):
            res = authenticate_user(email, password)
            if res == "unavailable":
                # Told apart from wrong credentials on purpose: the old message
                # covered both, so a user whose database was down went off to
                # reset a password that was never wrong.
                st.error("⚠️ Could not reach the database. This is not a problem "
                         "with your password — please try again shortly.")
            elif res:
                st.session_state.is_logged_in = True
                st.session_state.user_id, st.session_state.user_name = res[0], res[1]
                st.toast("Logged in successfully! 🎉")
                navigate_to("Dashboard")
            else:
                st.error("Invalid email or password.")

    with tab2:
        new_name = st.text_input("Full Name", key="reg_name")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Create Account", key="btn_reg_submit"):
            if new_name and new_email and new_password:
                ok, reason = create_user(new_name, new_email, new_password)
                if ok:
                    st.success("Account created successfully! Switch to the Login tab now.")
                elif reason == "duplicate":
                    st.error("This email is already registered.")
                else:
                    # Not the user's fault, and not something they can fix by
                    # trying another email — say so instead of blaming them.
                    st.error("⚠️ Could not reach the database. Your account was "
                             "NOT created. Please try again shortly.")
            else:
                st.error("Please fill in all the fields.")

    st.markdown(f"<div style='text-align: center; margin-top: 15px; color:{_muted};'>Or continue with a global account</div>", unsafe_allow_html=True)
    st.button("🌐 Central Authorization (Google)")
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- PAGE: USER DASHBOARD (DYNAMIC WITH MYSQL) ---
def render_dashboard():
    st.markdown(f"<h1 class='gradient-text'>Strategic Dashboard | Welcome, {st.session_state.user_name} 👋</h1>", unsafe_allow_html=True)

    history, history_ok = get_user_interview_history(st.session_state.user_id)
    if not history_ok:
        st.warning("⚠️ Could not read your history — the figures below are not "
                   "your real data.")
    total_interviews = len(history)
    valid_scores = [item['score'] for item in history if item.get('score') is not None]
    avg_score = int(sum(valid_scores) / len(valid_scores)) if len(valid_scores) > 0 else 0

    # Change since the previous session — the single number a returning
    # candidate actually looks for. Shown only from the second session on,
    # because "+0%" against nothing is noise.
    delta_txt = ""
    if len(valid_scores) >= 2:
        delta = valid_scores[-1] - valid_scores[-2]
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        colour = _c_green if delta > 0 else (_c_red if delta < 0 else _muted)
        delta_txt = (f"<div style='color:{colour}; font-size:0.85rem;'>"
                     f"{arrow} {abs(delta)}% since last session</div>")

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f"<div class='premium-card'><small style='color: {_muted};'>Total Interviews</small><h2 style='color:#7C3AED;'>{total_interviews}</h2></div>", unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='premium-card'><small style='color: {_muted};'>Latest Score</small><h2 style='color:{_c_cyan};'>{valid_scores[-1] if valid_scores else 0}%</h2>{delta_txt}<small style='color:{_muted_soft};'>average {avg_score}%</small></div>", unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='premium-card'><small style='color: {_muted};'>CV Status</small><h2 style='color:{_c_green};'>{'Analyzed & Uploaded' if st.session_state.cv_uploaded else 'Not uploaded yet'}</h2></div>", unsafe_allow_html=True)
    with m4: st.markdown(f"<div class='premium-card'><small style='color: {_muted};'>Current Plan</small><h2 style='color:{_c_amber};'>Pro Plan</h2></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("<div class='premium-card'><h3>📈 Technical Performance Progress Curve</h3>", unsafe_allow_html=True)
        if total_interviews > 0:
            df = pd.DataFrame(history)
            df['session_date'] = pd.to_datetime(df['session_date']).dt.strftime('%m/%d %H:%M')
            st.line_chart(data=df, x='session_date', y='score', use_container_width=True)
            for index, row in df.iterrows():
                with st.expander(f"💼 {row.get('job_title')} - Score: {row.get('score')}%"):
                    st.info(row.get('ai_evaluation') or "No text report available.")
        else:
            st.info("You haven't done any interviews yet to generate the chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        progress, prog_ok = get_skill_progress(st.session_state.user_id)
        multi = {k: v for k, v in progress.items() if len(v) >= 2}
        if prog_ok and multi:
            improved, declined, stuck = [], [], []
            for skill, points in multi.items():
                first, last = points[0][1], points[-1][1]
                change = (last - first) * 100
                row = (skill, first * 100, last * 100, change)
                if change >= 5:
                    improved.append(row)
                elif change <= -5:
                    declined.append(row)
                elif last < 0.5:
                    stuck.append(row)

            body = ""
            for title, group, colour, sign in (
                ("تحسّنت / Improved", sorted(improved, key=lambda r: -r[3]), _c_green, "+"),
                ("تراجعت / Declined", sorted(declined, key=lambda r: r[3]), _c_red, ""),
                ("ما زالت ضعيفة / Still weak", stuck, _c_yellow, ""),
            ):
                if not group:
                    continue
                body += f"<p style='color:{colour}; margin:8px 0 2px;'><b>{title}</b></p>"
                for skill, first, last, change in group[:4]:
                    body += (f"<p style='margin:2px 0; font-size:0.85rem;'>• {skill}: "
                             f"{first:.0f}% → <b>{last:.0f}%</b> "
                             f"<span style='color:{colour};'>({sign}{change:.0f}%)</span></p>")
            st.markdown(
                f"<div class='premium-card'><h3>📊 Skill Progress</h3>{body}"
                f"<p style='color:{_muted_soft}; font-size:0.75rem; margin-top:10px;'>"
                f"Compared across your sessions, first to latest.</p></div>",
                unsafe_allow_html=True)
        elif prog_ok and progress:
            st.markdown(
                "<div class='premium-card'><h3>📊 Skill Progress</h3>"
                f"<p style='color:{_muted}; font-size:0.9rem;'>Complete a second "
                "interview on the same skills to see how much you improved.</p></div>",
                unsafe_allow_html=True)

        # Recommendations are the evaluator's own feedback on the weakest
        # answers, so they refer to what this candidate actually said rather
        # than to a fixed sentence. Session state is preferred only while an
        # interview is actually running in this tab, since it holds answers
        # not yet written; otherwise the stored evaluations are read back, so
        # the advice survives logging out.
        rated = [a for a in st.session_state.interview_answers
                 if isinstance(a.get("score"), (int, float)) and a.get("feedback")]
        rec_ok = True
        if not rated:
            rated, rec_ok = get_weakest_answers(st.session_state.user_id)
        weakest = sorted(rated, key=lambda a: a["score"])[:3]
        if weakest:
            items = "<br><br>".join(
                f"• <b>{a.get('targets_skill') or 'General'}</b> "
                f"({a['score'] * 100:.0f}%) — {a['feedback']}" for a in weakest)
        elif not rec_ok:
            items = ("Could not read your past evaluations — the database is "
                     "unreachable right now.")
        else:
            items = "Complete an interview to see recommendations based on your own answers."
        st.markdown(
            '<div class="premium-card"><h3>🤖 Personalized AI Recommendations</h3>'
            f'<p style="color: {_muted}; font-size: 0.9rem; line-height: 1.6;">'
            f'{items}</p></div>', unsafe_allow_html=True)

def _rtl(text: str) -> str:
    """Wrap generated text so Arabic renders right-to-left.

    Applied per string rather than per page, because the interface labels stay
    English while the questions, feedback and reasoning are in the candidate's
    language.
    """
    if st.session_state.get("interview_language") != "ar":
        return text
    return f"<span class='rtl' style='display:block;'>{text}</span>"


def _skill_verdict(avg: float, untested: bool = False) -> tuple:
    """Colour and wording for one skill in the final report.

    The two cut-offs are imported from ai_engine rather than chosen here, and
    they are the same ones the agent routes on during the interview: below
    ANSWER_WEAK_THRESHOLD it abandons the skill, above ANSWER_STRONG_THRESHOLD
    it probes deeper. Reusing them means the report cannot call a skill
    "solid" that the agent walked away from mid-interview — the candidate
    reads the same judgement the system acted on.

    `untested` is separate from any score. Every answer on this skill missed
    the topic, so nothing was learned about it — and "we did not find out" must
    not be printed as "weak", which is a claim about the candidate.
    """
    if untested:
        return _muted, "not tested — the answer did not address it"
    if avg < ANSWER_WEAK_THRESHOLD:
        return _c_red, "weak — needs work"
    if avg > ANSWER_STRONG_THRESHOLD:
        return _c_green, "strong"
    return _c_yellow, "adequate"


# Snippets the toolbar inserts. Each is a Markdown construct the preview and
# the final report both render, so what the candidate formats is what everyone
# later reads.
# ==========================================
#  Answer integrity
# ==========================================
# Two independent signals, deliberately of different kinds.
#
# 1. Typing rate — measured on the SERVER from the moment the question is
#    rendered to the moment the answer is submitted. It cannot be faked from
#    the browser, and it is decisive: the world record for sustained copy
#    typing is about 212 wpm, while composing an answer while thinking rarely
#    exceeds 40-60. A 500-word answer submitted in 15 seconds is 2000 wpm,
#    which is not a fast typist.
#
# 2. Focus loss — counted in the browser. Weaker evidence (a notification can
#    steal focus innocently) and defeatable by a second device, so it is
#    reported as context rather than used as proof. Its real value is
#    deterrence: candidates are told it is recorded.
_WPM_SUSPICIOUS = 120      # above careful composition, still humanly possible
_WPM_IMPOSSIBLE = 200      # past the sustained human record — pasted
_MIN_WORDS_TO_JUDGE = 40   # short answers give too little signal to rate


def _integrity_check(text: str, seconds: float, focus_losses: int) -> dict:
    """Score how plausible it is that this answer was actually typed here."""
    words = len(text.split())
    wpm = (words / (seconds / 60)) if seconds > 0 else 0.0

    if words < _MIN_WORDS_TO_JUDGE:
        verdict = "too_short_to_judge"
    elif wpm >= _WPM_IMPOSSIBLE:
        verdict = "pasted"
    elif wpm >= _WPM_SUSPICIOUS:
        verdict = "suspicious"
    else:
        verdict = "plausible"

    return {
        "seconds": round(seconds, 1),
        "words": words,
        "wpm": round(wpm),
        "focus_losses": focus_losses,
        "verdict": verdict,
    }


def _focus_tracker(idx: int) -> None:
    """Count how many times the candidate leaves this tab.

    Streamlit's components.html runs in a sandboxed iframe with no channel
    back to Python, so the count is written into the parent page's query
    string; the next rerun — which the submit button causes — reads it. Best
    effort by nature: it cannot see a second screen or a phone, which is
    exactly why the typing rate above carries the real weight.
    """
    components.html(
        f"""<script>
        (function () {{
          const doc = window.parent.document;
          const KEY = 'focus_lost_{idx}';
          if (doc.__introTracker === KEY) return;   // Streamlit re-runs constantly
          doc.__introTracker = KEY;
          let lost = 0;
          const push = () => {{
            const u = new URL(window.parent.location);
            u.searchParams.set('fl', lost);
            window.parent.history.replaceState({{}}, '', u);
          }};
          window.parent.addEventListener('blur', () => {{ lost++; push(); }});
          doc.addEventListener('visibilitychange', () => {{
            if (doc.hidden) {{ lost++; push(); }}
          }});
        }})();
        </script>""",
        height=0,
    )


_EDITOR_SNIPPETS = [
    ("**B**", "**نص غليظ**", "**bold text**", "غليظ / bold"),
    ("*I*", "*نص مائل*", "*italic text*", "مائل / italic"),
    ("`</>`", "`كود`", "`code`", "كود قصير / inline code"),
    ("▤", "\n```python\nprint('hello')\n```\n", "\n```python\nprint('hello')\n```\n",
     "كتلة كود / code block"),
    ("•", "\n- عنصر أول\n- عنصر ثانٍ\n", "\n- first point\n- second point\n",
     "نقاط / bullets"),
    ("1.", "\n1. الخطوة الأولى\n2. الخطوة الثانية\n",
     "\n1. first step\n2. second step\n", "ترقيم / numbered"),
]


def _answer_editor(label: str, idx: int) -> str:
    """A formatting-aware answer box.

    Streamlit's text_area is plain text and gives no access to the caret, so a
    true WYSIWYG toolbar is not possible without a third-party component — and
    those are heavy, which matters on a 1 GB deployment. Instead the toolbar
    appends Markdown snippets the candidate then edits, and a live preview
    shows exactly how the answer will be rendered.

    The buttons must run BEFORE the text_area is created: Streamlit forbids
    writing to a widget's session_state key after that widget exists in the
    same run.
    """
    key = f"ans_{idx}"
    st.session_state.setdefault(key, "")
    is_ar = st.session_state.interview_language == "ar"

    # The clock starts when the question first appears, not when typing starts:
    # the gap between the two is thinking time, which is exactly what a pasted
    # answer skips.
    st.session_state.setdefault(f"t0_{idx}", time.time())
    _focus_tracker(idx)

    cols = st.columns(len(_EDITOR_SNIPPETS) + 1)
    for col, (face, ar_text, en_text, tip) in zip(cols, _EDITOR_SNIPPETS):
        if col.button(face, key=f"tb_{idx}_{face}", help=tip,
                      use_container_width=True):
            snippet = ar_text if is_ar else en_text
            current = st.session_state[key]
            joiner = "" if (not current or current.endswith(("\n", " "))) else " "
            st.session_state[key] = current + joiner + snippet
            st.rerun()

    show_preview = cols[-1].toggle("👁", key=f"pv_{idx}",
                                   help="معاينة / preview")

    text = st.text_area(label, key=key, height=160,
                        placeholder=("يمكنك استخدام **غليظ** و `كود` و - نقاط"
                                     if is_ar else
                                     "You can use **bold**, `code` and - bullets"))

    if show_preview and text.strip():
        st.caption("المعاينة / Preview")
        st.markdown(_rtl(text) if is_ar else text, unsafe_allow_html=is_ar)

    elapsed = time.time() - st.session_state[f"t0_{idx}"]
    words = len(text.split())
    mins, secs = divmod(int(elapsed), 60)
    # Stated openly. A hidden check is surveillance; a declared one is a rule,
    # and telling candidates it exists is most of its deterrent value.
    st.caption(
        f"⏱️ {mins}:{secs:02d}  ·  {words} "
        + ("كلمة  ·  الوقت وتبديل النوافذ مُسجَّلان لضمان نزاهة التدريب"
           if is_ar else
           "words  ·  time and tab switches are recorded for training integrity")
    )
    return text


def _reset_interview_state():
    """Clear everything the live interview accumulates.

    The queue, the evaluations and the agent's budget counters all carry over
    between Streamlit reruns by design; without an explicit reset a second
    interview would inherit the first one's follow-ups and scores.
    """
    st.session_state.interview_queue = []
    st.session_state.interview_evaluations = []
    st.session_state.covered_skills = []
    st.session_state.probe_depth = 0
    st.session_state.off_plan_used = 0
    st.session_state.final_score = None


def _extract_pdf_text(uploaded_file) -> str:
    # Normalised on the way in: Arabic PDFs return presentation-form glyphs
    # rather than base letters, which look right but compare wrong. See
    # normalize_text() in ai_engine.py.
    reader = PdfReader(uploaded_file)
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return normalize_text(raw)

# --- PAGE: UPLOAD SYSTEM (CONNECTED TO MYSQL) ---
def render_upload():
    st.markdown("<h1 class='gradient-text'>Cloud Data Upload & Analysis System</h1>", unsafe_allow_html=True)

    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("1. Upload Your CV")
    uploaded_file = st.file_uploader("Choose your CV file (PDF format)", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.cv_file_name:
        with st.spinner("Analyzing your CV with AI..."):
            cv_text = _extract_pdf_text(uploaded_file)
            result = extract_skills(cv_text, document_type="CV")

        if not cv_text.strip():
            st.session_state.cv_uploaded = False
            st.session_state.cv_skills = {}
            st.error("Could not extract any text from the file. Make sure the PDF contains real text, not a scanned image.")
        elif "error" in result:
            st.session_state.cv_uploaded = False
            st.session_state.cv_skills = {}
            st.error(f"Could not analyze the CV: {result['error']}")
        else:
            total = len(result["technical_skills"]) + len(result["soft_skills"]) + len(result["languages"])
            st.session_state.cv_uploaded = True
            st.session_state.cv_skills = result
            st.session_state.cv_file_name = uploaded_file.name
            # The CV's language decides the interview's language. Detected here,
            # once, from the text — before it is discarded and only the
            # extracted skills are kept in session state.
            st.session_state.interview_language = detect_language(cv_text)
            st.success(f"CV analyzed successfully — {total} skills extracted.")
            if st.session_state.interview_language == "ar":
                st.info("تم اكتشاف سيرة ذاتية بالعربية — ستكون المقابلة والتقييم بالعربية.")

    if st.session_state.cv_skills:
        tech = st.session_state.cv_skills.get("technical_skills", [])
        soft = st.session_state.cv_skills.get("soft_skills", [])
        langs = st.session_state.cv_skills.get("languages", [])
        with st.expander(f"View extracted skills ({len(tech) + len(soft) + len(langs)})", expanded=True):
            if tech:
                st.markdown("**Technical Skills**")
                for skill in tech:
                    st.markdown(f"- {skill}")
            if soft:
                st.markdown("**Soft Skills**")
                for skill in soft:
                    st.markdown(f"- {skill}")
            if langs:
                st.markdown("**Languages**")
                for lang in langs:
                    st.markdown(f"- {lang}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("2. Target Job Details")
    job_title = st.text_input("Target Job Title", "Software Engineer")
    # Placeholder rather than a pre-filled value: the old default made the user
    # delete a sentence before typing, and it had to be string-matched below to
    # tell "untouched" from "empty".
    #
    # The hint is not only politeness. Salary, benefits, working hours, location
    # and how-to-apply carry no skills, so they add prompt tokens and give the
    # model more irrelevant text to reason over — a full posting is what pushed
    # one request into spending its entire budget deliberating without ever
    # writing an answer. Trimming shrinks every call, not just the failing ones.
    job_desc = st.text_area(
        "Enter the Job Description",
        placeholder=("Paste the responsibilities and requirements.\n\n"
                     "You can leave out salary, benefits, working hours, "
                     "location and how-to-apply — they hold no skills, and "
                     "dropping them makes the analysis faster and more accurate."),
        height=170,
        help="Only the responsibilities and requirements are read. Everything "
             "else in a posting is ignored, so pasting it just slows the "
             "analysis down.",
    )

    col1, col2 = st.columns(2)
    with col1: experience = st.selectbox("Current Experience Level", ["Junior (0-2 years)", "Mid-Level (2-5 years)", "Senior (5+ years)"])
    with col2:
        # Detection from the CV supplies the default; the candidate decides.
        # An English CV does not mean an English interview — in this market a
        # candidate with an English technical CV is very often interviewed in
        # Arabic, and the supervisor's stated reason for Arabic support is
        # reaching sectors outside tech at all.
        _lang_options = ["English (Professional)", "Arabic / العربية"]
        _detected_idx = 1 if st.session_state.interview_language == "ar" else 0
        lang_level = st.selectbox(
            "Interview Language", _lang_options, index=_detected_idx,
            help="Detected from your CV. Change it if you want to be interviewed "
                 "in the other language.",
        )
        st.session_state.interview_language = "ar" if lang_level.startswith("Arabic") else "en"

    if st.button("🔍 Analyze the Gap Between Your Skills and the Job"):
        if not st.session_state.cv_skills:
            st.warning("Upload your CV and analyze your skills first (step 1) before running the gap analysis.")
        elif not job_desc.strip():
            st.warning("Enter the actual job description first.")
        else:
            with st.spinner("Analyzing the job requirements and comparing them to your skills..."):
                pipeline_result = run_cv_jd_pipeline(
                    jd_text=job_desc,
                    cv_skills=st.session_state.cv_skills,
                    language=st.session_state.interview_language,
                )
                if pipeline_result.get("error"):
                    st.session_state.skill_gap = {}
                    st.session_state.generated_questions = []
                    st.error(f"Could not analyze the job description: {pipeline_result['error']}")
                else:
                    st.session_state.skill_gap = pipeline_result["skill_gap"]
                    # Needed later by the agent to build the skill vocabulary
                    # it scores answers against.
                    st.session_state.jd_skills = pipeline_result.get("jd_skills", {})
                    st.session_state.generated_questions = pipeline_result.get("questions", {}).get("questions", [])

    if st.session_state.skill_gap:
        gap = st.session_state.skill_gap
        missing_tech = gap.get("missing_technical_skills", [])
        missing_soft = gap.get("missing_soft_skills", [])
        missing_langs = gap.get("missing_languages", [])
        total_missing = len(missing_tech) + len(missing_soft) + len(missing_langs)
        with st.expander(f"Gap analysis result ({total_missing} missing skills)", expanded=True):
            if total_missing == 0:
                st.success("Your skills cover all the requirements visible in the job description! 🎉")
            else:
                if missing_tech:
                    st.markdown("**Required technical skills you're missing**")
                    for s in missing_tech:
                        st.markdown(f"- {s}")
                if missing_soft:
                    st.markdown("**Required soft skills you're missing**")
                    for s in missing_soft:
                        st.markdown(f"- {s}")
                if missing_langs:
                    st.markdown("**Required languages you're missing**")
                    for s in missing_langs:
                        st.markdown(f"- {s}")

    if st.session_state.generated_questions:
        st.info(
            f"🎯 {len(st.session_state.generated_questions)} gap-prioritized interview questions are ready. "
            "Start the session below to answer them in the Interview Simulator."
        )

    if st.button("Save Data & Generate the AI Interview Engine"):
        if not job_title.strip() or not job_desc.strip():
            st.warning("Please make sure all job detail fields are filled in.")
        elif not st.session_state.cv_skills:
            st.warning("Upload and analyze your CV first (step 1).")
        else:
            if not st.session_state.generated_questions:
                with st.spinner("Generating gap-prioritized interview questions..."):
                    pipeline_result = run_cv_jd_pipeline(
                    jd_text=job_desc,
                    cv_skills=st.session_state.cv_skills,
                    language=st.session_state.interview_language,
                )
                    if pipeline_result.get("error"):
                        st.error(f"Could not generate the interview questions: {pipeline_result['error']}")
                        st.stop()
                    st.session_state.skill_gap = pipeline_result["skill_gap"]
                    # Needed later by the agent to build the skill vocabulary
                    # it scores answers against.
                    st.session_state.jd_skills = pipeline_result.get("jd_skills", {})
                    st.session_state.generated_questions = pipeline_result.get("questions", {}).get("questions", [])

            with st.spinner("Initializing the AI interview server in the database..."):
                int_id = create_interview_session(st.session_state.user_id, job_title, experience, job_desc)
                if int_id:
                    st.session_state.current_interview_id = int_id
                    st.session_state.interview_started = True
                    st.session_state.current_question = 0
                    st.session_state.interview_answers = []
                    _reset_interview_state()
                    st.session_state.interview_queue = [
                        dict(q) for q in st.session_state.generated_questions]
                    save_interview_progress(int_id)
                    st.toast("Session and engine built successfully! 🚀")
                    navigate_to("Interview Engine")
                else:
                    st.error(f"⚠️ Could not create the interview session — "
                             f"database unreachable ({_DB_ERROR}). Nothing was saved.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_db_badge():
    """A visible, honest indicator of whether anything is being saved.

    Placed in the sidebar because a silent database is invisible otherwise —
    the interview looks identical whether or not a single row is written.
    """
    ok, detail = db_status()
    if ok:
        st.sidebar.caption(f"🟢 Database connected · {detail}")
    else:
        st.sidebar.error("🔴 Database NOT connected — nothing is being saved.")
        st.sidebar.caption(detail)


def render_interview():
    st.markdown("<h1 class='gradient-text'>🤖 Live AI Interview Simulator</h1>", unsafe_allow_html=True)

    # The answer box is the one input the candidate types Arabic into, so it
    # is flipped only while an Arabic interview is running. Injected here
    # rather than in the global stylesheet because the language is chosen
    # after that stylesheet has already been written.
    if st.session_state.interview_language == "ar":
        st.markdown(
            "<style>[data-testid='stTextArea'] textarea"
            "{direction: rtl; text-align: right;}</style>",
            unsafe_allow_html=True)

    if not st.session_state.interview_started or st.session_state.current_interview_id is None:
        # This is where a candidate lands after logging back in, so it is where
        # an interrupted interview has to be offered back to them. Anything
        # that clears session_state — a logout, a browser refresh, the host
        # putting the app to sleep — brings them here mid-interview.
        pending_row, pending_state = load_unfinished_interview(st.session_state.user_id)
        if pending_row and pending_state:
            answered = len(pending_state.get("interview_answers") or [])
            total = len(pending_state.get("interview_queue") or [])
            st.markdown(
                f"<div class='premium-card' style='border-left: 4px solid {_c_amber};'>"
                f"<h3>⏸️ You have an unfinished interview</h3>"
                f"<p><b>{pending_row['job_title']}</b> — "
                f"{answered} of {total} questions answered, started "
                f"{pending_row['session_date']:%d %b %Y at %H:%M}.</p>"
                f"<p style='color:{_muted}; font-size:0.9rem;'>Your answers were "
                f"saved as you submitted them. Continuing picks up at the next "
                f"question; starting fresh leaves this session scored as it "
                f"stands.</p></div>",
                unsafe_allow_html=True)
            resume_col, fresh_col = st.columns(2)
            with resume_col:
                if st.button("▶️ Continue where I left off", key="btn_resume"):
                    for field, value in pending_state.items():
                        st.session_state[field] = value
                    st.session_state.current_interview_id = pending_row["id"]
                    st.session_state.interview_started = True
                    st.session_state.cv_uploaded = bool(pending_state.get("cv_skills"))
                    st.rerun()
            with fresh_col:
                if st.button("🆕 Start a new interview instead", key="btn_fresh"):
                    navigate_to("Upload System")

        st.markdown("<div class='premium-card' style='text-align: center; padding: 40px;'>", unsafe_allow_html=True)
        st.write("The engine is waiting for data setup from the Upload System page.")
        if st.button("Go to the Upload page now"):
            navigate_to("Upload System")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # The queue starts as the generated plan, then the agent reshapes it as
        # the interview runs. Seeded once so an in-progress interview survives
        # Streamlit's reruns.
        if not st.session_state.interview_queue:
            st.session_state.interview_queue = [
                dict(q) for q in st.session_state.generated_questions
            ] or [{"question": t, "targets_skill": "", "is_soft_skill": False}
                  for t in st.session_state.mock_questions]

        queue = st.session_state.interview_queue
        curr_idx = st.session_state.current_question
        total_qs = len(queue)

        if curr_idx < total_qs:
            current_q = queue[curr_idx]
            current_q_text = current_q["question"]
            is_last = curr_idx >= total_qs - 1

            st.markdown(
                f"<div class='premium-card' style='border-left: 4px solid #7C3AED;'>"
                f"<h5>Current Question ({curr_idx + 1} of {total_qs}):</h5>"
                f"<h3 style='color: {_c_violet};'>{_rtl(current_q_text)}</h3></div>",
                unsafe_allow_html=True,
            )
            answer_label = ("اكتب إجابتك الكاملة هنا:"
                            if st.session_state.interview_language == "ar"
                            else "Enter your full answer here:")
            user_ans = _answer_editor(answer_label, curr_idx)

            def _process_answer():
                """Evaluate the answer, then let the agent decide what comes next.

                This is where the interview stops being a fixed list: the agent
                may append a follow-up it just generated, or drop the remaining
                questions on a skill the candidate clearly does not have.
                """
                state = {
                    "remaining_skills": [q.get("targets_skill", "")
                                         for q in queue[curr_idx + 1:]],
                    "covered_skills": st.session_state.covered_skills,
                    "asked_count": curr_idx + 1,
                    "budget_left": max(0, total_qs - curr_idx - 1),
                    "probe_depth": st.session_state.probe_depth,
                    "off_plan_used": st.session_state.off_plan_used,
                }
                cycle = run_answer_cycle(
                    current_q, user_ans,
                    st.session_state.cv_skills, st.session_state.get("jd_skills", {}),
                    state, language=st.session_state.interview_language,
                )

                evaluation = cycle.get("evaluation", {})
                score = evaluation.get("final_score")
                feedback = evaluation.get("feedback", "")

                # Measured before anything else touches the clock.
                integrity = _integrity_check(
                    user_ans,
                    time.time() - st.session_state.get(f"t0_{curr_idx}", time.time()),
                    int(st.query_params.get("fl", 0) or 0),
                )
                evaluation = dict(evaluation)
                evaluation["integrity"] = integrity
                # Stored with the answer so progress can later be reported per
                # skill. The column holds only the question text, and matching
                # a skill back out of that text would be guesswork.
                evaluation["targets_skill"] = current_q.get("targets_skill", "")
                # The agent's decision, stored with the answer for the same
                # reason targets_skill is. These four lived only in
                # st.session_state, so they were shown in the report and then
                # lost when the browser session ended: two completed interviews
                # exist in the database with no record of what the agent chose
                # at any point, and whether a probe happened had to be inferred
                # from the wording of the next question.
                #
                # used_fallback matters most. It was added to measure how often
                # the model replies without invoking a tool — a number that
                # belongs in the results chapter — and it could never be
                # computed, because nothing wrote it down.
                evaluation["route"] = cycle.get("route")
                evaluation["reason"] = cycle.get("reason", "")
                evaluation["thought"] = (cycle.get("thought") or "")[:1200]
                evaluation["used_fallback"] = cycle.get("used_fallback")
                evaluation["budget_capped"] = cycle.get("budget_capped")

                save_question_answer(st.session_state.current_interview_id,
                                     current_q_text, user_ans,
                                     score=score, evaluation=evaluation)
                st.session_state.interview_answers.append({
                    "question": current_q_text,
                    "answer": user_ans,
                    "targets_skill": current_q.get("targets_skill", ""),
                    "score": score,
                    "feedback": feedback,
                    "skill_untested": evaluation.get("skill_untested", False),
                    "route": cycle.get("route"),
                    "reason": cycle.get("reason", ""),
                    "thought": cycle.get("thought", ""),
                    "used_fallback": cycle.get("used_fallback"),
                    "integrity": integrity,
                })
                if "error" not in cycle:
                    st.session_state.interview_evaluations.append(evaluation)

                # --- act on the agent's decision -------------------------
                route = cycle.get("route")
                new_q = cycle.get("new_question")
                if route == "probe" and new_q:
                    queue.insert(curr_idx + 1, new_q)
                    st.session_state.probe_depth += 1
                elif route == "ask_about_skill" and new_q:
                    queue.insert(curr_idx + 1, new_q)
                    st.session_state.off_plan_used += 1
                    st.session_state.probe_depth = 0
                else:
                    st.session_state.probe_depth = 0
                    skill = current_q.get("targets_skill", "")
                    if skill and skill not in st.session_state.covered_skills:
                        st.session_state.covered_skills.append(skill)
                    if route == "skip_skill" and skill:
                        # Drop the rest of this skill's questions rather than
                        # pressing on a topic already shown to be a dead end.
                        st.session_state.interview_queue = (
                            queue[:curr_idx + 1]
                            + [q for q in queue[curr_idx + 1:]
                               if q.get("targets_skill") != skill]
                        )
                # The snapshot goes last, so it records the queue as the
                # agent has just reshaped it rather than as it was before.
                save_interview_progress(st.session_state.current_interview_id)
                return cycle

            if not is_last:
                if st.button("Submit Answer & Go to Next Question ➡️"):
                    with st.spinner("Evaluating your answer..."):
                        cycle = _process_answer()
                    if cycle.get("error"):
                        st.error(cycle["error"])
                    else:
                        st.session_state.current_question += 1
                        st.rerun()
            else:
                if st.button("Finish Interview & Get Results 🎓"):
                    with st.spinner("Evaluating your final answer..."):
                        cycle = _process_answer()
                    if cycle.get("error"):
                        st.error(cycle["error"])
                    else:
                        summary = " ".join(
                            a["feedback"] for a in st.session_state.interview_answers
                            if a.get("feedback"))[:2000]
                        # The score comes back from the database rather than
                        # being computed here, so what is stored and what is
                        # shown are the same number by construction.
                        st.session_state.final_score = finalize_interview(
                            st.session_state.current_interview_id, summary)
                        st.session_state.current_question += 1
                        st.rerun()
        else:
            st.markdown(f"<div class='premium-card' style='background: rgba(52, 211, 153, 0.03); border-color: {_c_green};'><h3>🎉 Interview completed successfully! Generating your in-depth evaluation results...</h3></div>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.003)
                progress_bar.progress(percent_complete + 1)

            st.markdown("<h2 class='gradient-text'>Final Report & Technical Performance Analysis</h2>", unsafe_allow_html=True)

            answers = st.session_state.interview_answers
            evals = st.session_state.interview_evaluations
            scores = [e.get("final_score") for e in evals
                      if isinstance(e.get("final_score"), (int, float))]
            # Prefer the figure finalize_interview() computed from the stored
            # answer rows, so the report shows exactly what was written. The
            # session-state average is only a fallback for a report re-rendered
            # after the write already happened.
            overall = st.session_state.get("final_score")
            if overall is None:
                overall = round(100 * sum(scores) / len(scores)) if scores else 0

            # Colour and label follow the same thresholds the agent routes on,
            # so what the candidate reads matches what the system decided.
            if overall >= 80:
                colour, verdict = _c_green, "Ready for the real interview"
            elif overall >= 40:
                colour, verdict = _c_yellow, "Solid in places — see the weak skills below"
            else:
                colour, verdict = _c_red, "Needs preparation before interviewing"

            score_col1, score_col2 = st.columns([1, 2])
            with score_col1:
                st.markdown(
                    f"<div class='premium-card' style='text-align: center;'><h4>Overall Score</h4>"
                    f"<h1 style='font-size: 4rem; color: {colour};'>{overall}%</h1>"
                    f"<span class='badge'>{verdict}</span>"
                    f"<p style='color:{_muted}; font-size:0.8rem; margin-top:8px;'>"
                    f"mean of {len(scores)} evaluated answer(s)</p></div>",
                    unsafe_allow_html=True,
                )
            with score_col2:
                # Per-skill, worst first — the point of the report is what to
                # go and fix, not a compliment.
                by_skill = {}
                # A skill counts as untested only when EVERY answer on it missed
                # the topic. One off-topic answer among several still leaves
                # evidence, so the average stands.
                untested_skill = {}
                for a in answers:
                    if isinstance(a.get("score"), (int, float)):
                        key = a.get("targets_skill") or "general"
                        by_skill.setdefault(key, []).append(a["score"])
                        untested_skill[key] = (untested_skill.get(key, True)
                                               and bool(a.get("skill_untested")))
                rows = sorted(((s, sum(v) / len(v)) for s, v in by_skill.items()),
                              key=lambda x: x[1])
                body = "".join(
                    "<p><b>• {}:</b> {:.0f}% &nbsp;<span style='color:{};"
                    "font-size:0.85rem;'>{}</span></p>".format(
                        skill, avg * 100, *_skill_verdict(avg, untested_skill.get(skill, False)))
                    for skill, avg in rows
                ) or f"<p style='color:{_muted};'>No answers were evaluated.</p>"
                st.markdown(
                    f"<div class='premium-card'><h4>Skill Breakdown "
                    f"<span style='color:{_muted}; font-size:0.8rem;'>(weakest first)</span></h4>"
                    f"{body}</div>",
                    unsafe_allow_html=True,
                )

            # --- what to work on next ------------------------------------
            # The evaluator already wrote feedback on every answer, but it sat
            # inside the collapsed transcript below: advice nobody opens is
            # advice nobody reads. This lifts it out for the skills that did
            # not reach the strong threshold, worst first, quoting the
            # evaluator on the weakest answer for each skill rather than
            # inventing generic study tips.
            worst_answer = {}
            for a in answers:
                skill = a.get("targets_skill") or "general"
                if not a.get("feedback") or not isinstance(a.get("score"), (int, float)):
                    continue
                if skill not in worst_answer or a["score"] < worst_answer[skill]["score"]:
                    worst_answer[skill] = a
            todo = [(skill, avg) for skill, avg in rows
                    if avg <= ANSWER_STRONG_THRESHOLD and skill in worst_answer][:4]
            if todo:
                items = ""
                for skill, avg in todo:
                    colour, label = _skill_verdict(avg, untested_skill.get(skill, False))
                    items += (
                        f"<p style='margin:14px 0 4px;'><b>• {skill}</b> "
                        f"<span style='color:{colour}; font-size:0.85rem;'>"
                        f"{avg * 100:.0f}% {label}</span></p>"
                        f"<p style='color:{_muted}; font-size:0.9rem; "
                        f"margin:0 0 0 14px;'>{_rtl(worst_answer[skill]['feedback'])}</p>")
                st.markdown(
                    f"<div class='premium-card'><h4>🎯 What to work on next "
                    f"<span style='color:{_muted}; font-size:0.8rem;'>"
                    f"(the evaluator's own notes on your weakest answer per skill)"
                    f"</span></h4>{items}</div>",
                    unsafe_allow_html=True,
                )
            elif rows:
                st.markdown(
                    f"<div class='premium-card'><h4>🎯 What to work on next</h4>"
                    f"<p style='color:{_muted};'>Every skill tested came out above "
                    f"{ANSWER_STRONG_THRESHOLD * 100:.0f}% — nothing stood out as "
                    f"needing work in this session.</p></div>",
                    unsafe_allow_html=True,
                )

            if answers:
                with st.expander(f"📝 Session transcript ({len(answers)} answers)", expanded=False):
                    for i, item in enumerate(answers, start=1):
                        score = item.get("score")
                        score_txt = f" — {score * 100:.0f}%" if isinstance(score, (int, float)) else ""
                        st.markdown(
                            f"<b>Q{i}.</b>{score_txt}{_rtl(item['question'])}",
                            unsafe_allow_html=True)
                        # Rendered as Markdown, not escaped: the candidate may
                        # have formatted the answer with the editor toolbar, and
                        # showing raw ** and ` symbols in the report would make
                        # a well-structured answer look worse than a plain one.
                        answer_md = item["answer"] or "_(no answer given)_"
                        if st.session_state.interview_language == "ar":
                            st.markdown(f"<div class='rtl'>", unsafe_allow_html=True)
                            st.markdown(answer_md)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(answer_md)
                        if item.get("feedback"):
                            st.markdown(f"💬 {_rtl(item['feedback'])}", unsafe_allow_html=True)

                        ig = item.get("integrity") or {}
                        if ig:
                            icon = {"pasted": "🔴", "suspicious": "🟠",
                                    "plausible": "🟢"}.get(ig["verdict"], "⚪")
                            note = ""
                            if ig["verdict"] == "pasted":
                                note = " — faster than any human types; this answer was pasted"
                            elif ig["verdict"] == "suspicious":
                                note = " — unusually fast for composed writing"
                            st.caption(
                                f"{icon} {ig['seconds']}s · {ig['words']} words · "
                                f"{ig['wpm']} wpm · left the tab {ig['focus_losses']}×{note}")
                        if item.get("thought"):
                            # The agent's written reasoning is what makes an
                            # adaptive decision auditable instead of silent.
                            st.markdown(
                                f"<details><summary style='color:#7C3AED; cursor:pointer;'>"
                                f"Why the interview went this way "
                                f"(<code>{item.get('route')}</code>)</summary>"
                                f"<p style='color:{_muted}; font-size:0.85rem;'>{item['thought'][:900]}</p>"
                                f"</details>",
                                unsafe_allow_html=True)
                        st.markdown("---")


            # The planned set, not the live queue: this panel reports on
            # question *generation*, so it must show what the generator
            # produced rather than what the agent later inserted or dropped.
            generated = st.session_state.generated_questions
            if generated:
                passed = sum(1 for q in generated if q.get("passes_gate"))
                gap_qs = sum(1 for q in generated if q.get("is_gap_skill"))
                existing_qs = len(generated) - gap_qs
                with st.expander(
                    f"⚙️ Question-generation diagnostics — "
                    f"coverage: {gap_qs} gap / {existing_qs} existing &nbsp;·&nbsp; "
                    f"Answerability Gate: {passed}/{len(generated)} passed",
                    expanded=False,
                ):
                    for i, q in enumerate(generated, start=1):
                        gate_icon = "✅" if q.get("passes_gate") else "⚠️"
                        gap_label = "gap skill" if q.get("is_gap_skill") else "existing skill"
                        
                        entities = q.get("content_entities")
                        entities_label = f"{entities:.2f}" if entities is not None else "n/a (soft skill)"
                        st.markdown(
                            f"**Q{i}** — targets <b>{q['targets_skill']}</b> ({gap_label}) &nbsp;|&nbsp; "
                            f"AS = <b>{q['answerability_score']:.2f}</b> {gate_icon} "
                            f"<span style='color:{_muted}; font-size:0.85rem;'>"
                            f"(entities {entities_label} · clarity {q['context_clarity']:.2f} · specificity {q['task_specificity']:.2f})"
                            f"</span>",
                            unsafe_allow_html=True,
                        )

            if st.button("Return to Dashboard & Track the Curve"):
                st.session_state.interview_started = False
                st.session_state.current_question = 0
                st.session_state.current_interview_id = None
                st.session_state.interview_answers = []
                _reset_interview_state()
                navigate_to("Dashboard")

def render_billing():
    st.markdown("<h1 class='gradient-text'>💳 Subscription Management & Global Billing System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='premium-card'><h3>Current Subscription Details</h3><p>You are currently subscribed to: <b>Pro Plan</b></p><p>Next renewal date: <b>June 30, 2026</b></p><span class='badge'>Status: Active via Stripe</span></div>", unsafe_allow_html=True)


st.sidebar.markdown("<h2 class='gradient-text' style='text-align:center;'>IntervAI Pro 🤖</h2>", unsafe_allow_html=True)
st.sidebar.toggle("🌙 Dark Mode", key="dark_mode")
st.sidebar.markdown(f"<hr style='border-color: {_hr};'>", unsafe_allow_html=True)
render_db_badge()

if not st.session_state.is_logged_in:
    available_pages = ["Home (Landing Page)", "Login Portal"]
    default_idx = 0 if st.session_state.page == "Landing" else 1

    nav_selection = st.sidebar.radio("Main Menu", available_pages, index=default_idx)
    st.session_state.page = "Landing" if nav_selection == "Home (Landing Page)" else "Auth"
else:
    st.sidebar.markdown(f"<p style='text-align:center; color:{_c_violet};'>👤 {st.session_state.user_name}</p>", unsafe_allow_html=True)
    menu_options = {
        "📊 Dashboard": "Dashboard",
        "📂 Upload System": "Upload System",
        "🎙️ Interview Simulator (AI Engine)": "Interview Engine",
        "💳 Subscriptions & Billing": "Billing",
    }

    if st.session_state.page not in menu_options.values():
        st.session_state.page = "Dashboard"

    list_values = list(menu_options.values())
    list_keys = list(menu_options.keys())
    default_idx = list_values.index(st.session_state.page)

    selection = st.sidebar.radio("Cloud Navigation Panel", list_keys, index=default_idx)
    st.session_state.page = menu_options[selection]

    st.sidebar.markdown(f"<hr style='border-color: {_hr};'>", unsafe_allow_html=True)
    if st.sidebar.button("Log Out"):
        st.session_state.is_logged_in = False
        st.session_state.user_id = None
        st.session_state.user_name = ""
        st.session_state.interview_started = False
        st.session_state.current_interview_id = None
        navigate_to("Landing")

if st.session_state.page == "Landing": render_landing()
elif st.session_state.page == "Auth": render_auth()
elif st.session_state.page == "Dashboard": render_dashboard()
elif st.session_state.page == "Upload System": render_upload()
elif st.session_state.page == "Interview Engine": render_interview()
elif st.session_state.page == "Billing": render_billing()
