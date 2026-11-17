# Put the app online for free (Streamlit Community Cloud)

Hugging Face now charges for Gradio Spaces, so we use **Streamlit Community Cloud** instead: free, runs the
same analysis, gives you a public link. You need a free **GitHub** account (where the code lives) and then
Streamlit reads it from there. About 20 minutes.

## 1. Put the code on GitHub
1. Go to **github.com** → **Sign up** (free) and verify your email.
2. Top right **+** → **New repository**.
   - Name: `cricket-ks-lab` · **Public** · leave everything else as it is · **Create repository**.
3. On the new empty page, click the link **"uploading an existing file"**.
4. Unzip `cricket-ks-lab.zip`. Open the `cricket-ks-lab` folder, select **everything inside it**
   (folders `ksd`, `docs`, `tests`, `.streamlit` and all files) and drag it into the GitHub page.
   - **Mac:** `.streamlit` is a hidden folder. In Finder press **Cmd + Shift + .** to show it.
     (If it's missing, the app still works - only the colour theme changes.)
5. Scroll down → **Commit changes**. Check that `streamlit_app.py`, `requirements.txt`, `packages.txt` and the
   `ksd` folder are listed.

## 2. Launch it on Streamlit
1. Go to **share.streamlit.io** → **Continue with GitHub** → allow access.
2. **Create app** → **Deploy a public app from GitHub** (or "Yup, I have an app").
3. Fill in:
   - Repository: `your-username/cricket-ks-lab`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: pick something like `cricket-kinematic-sequence`
4. Open **Advanced settings** → **Python version: 3.11** → Save.
5. **Deploy**. The first build takes 5–10 minutes (it installs MediaPipe). Then the app opens.
6. The first analysis downloads Google's pose model (~10 MB) automatically. A 5-second clip takes about
   20–60 seconds.

## 3. If something goes wrong
Click **Manage app** (bottom right of the app page) to see the logs. Look at the last red lines.

| What you see | Fix |
|---|---|
| `libGL.so.1` or `cv2` import error | Make sure `packages.txt` is in the repository, then **Reboot app** (Manage app → ⋮). |
| Error installing `mediapipe` | Check Python version is **3.11** (app settings → Advanced). Delete and redeploy if needed. |
| "Could not download the MediaPipe pose model" | Temporary network issue - Reboot app. |
| "No batter / no swing found" | The clip: whole body visible, front-on, trimmed to one shot. |
| Out of memory / app restarts | Trim the clip to 2–6 s. |

You can paste the log text to Claude and ask what to change. To change a file later: open it on GitHub → pencil
icon → edit → **Commit**; Streamlit updates the app by itself.

## 4. Good to know
- **Sleeping:** free apps sleep after a period without visitors. The next visitor sees a "wake up" button; it
  takes about a minute. Open the link yourself before any demo.
- **Optional Gemini phrasing:** Manage app → **Settings → Secrets**, add `GEMINI_API_KEY = "your-key"`
  (free key from Google AI Studio) and add a line `google-genai` to `requirements.txt` on GitHub. Gemini only
  re-words the summary; it never adds facts.

## Run on your own computer instead (optional)
1. Install Python 3.11 from python.org.
2. Open Terminal in the `cricket-ks-lab` folder and run:
   ```
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
3. The app opens in your browser.

(`app.py` is the same app built with Gradio. It works locally with `pip install gradio==5.49.1` then
`python app.py`, or on Hugging Face if you ever get a paid plan.)
