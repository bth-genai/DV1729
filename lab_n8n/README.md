# Lab: Building a Local AI Agent with n8n on Windows (WSL)

**Topic:** Low-code AI Automation & Orchestration  
**Duration:** Approx. 2 hours  

## 🎯 Objective
In this lab, you will set up a fully local AI automation environment. You will run **n8n** (workflow automation) inside **WSL**, and connect it to **Ollama** running locally on your machine.

By the end of this lab, you will have built an **"Intelligent Customer Feedback Classifier"** that:
1.  Reads incoming text.
2.  Uses a local LLM (e.g., Llama 3) to analyze sentiment.
3.  Drafts a response and routes the data automatically.

---

## 🛠 Part 1: Environment Setup

We will use WSL to run a Linux environment on your Windows machine, which is the industry standard for local web and AI development.

### Step 1: Prepare Ollama (Network Config)
By default, Ollama only listens to Windows. We need to allow it to listen to the Linux (WSL) system too.

1.  **Stop Ollama:**
    * Go to your taskbar (bottom right, near the clock).
    * Right-click the Ollama icon and select **Quit**.

2.  **Set Environment Variable (The GUI Way):**
    * Press the **Windows Key** and type: `env`
    * Select **"Edit environment variables for your account"**.
    * In the top section ("User variables"), click **New...**
    * **Variable name:** `OLLAMA_HOST`
    * **Variable value:** `0.0.0.0`
    * Click **OK**, then **OK** again to close the windows.

3.  **Expose Ollama to the network**
    * Click the Ollama icon in the taskbar and choose "Open Ollama"
    * Click Settings in the Ollama window that opens
    * Turn on the setting to "Expose Ollama to the network" 

3.  **Pull a model**
    Pull a fast model suitable for this lab (if you haven't already):
    ```powershell
    ollama pull llama3.2
    ```
    *(Note: You can also use `llama3.1`, `mistral`, or `phi3` depending on what you have installed).*
    
### Step 2: Enable WSL (Ubuntu)
*If you already have Ubuntu installed via WSL, you can skip to Step 2.*

1.  Open **PowerShell** as Administrator.
2.  Run the following command:
    ```powershell
    wsl --install
    ```
3.  **Restart your computer** when prompted.
4.  After restarting, open the application named **"Ubuntu"** from your Start menu.
5.  Follow the on-screen instructions to create a UNIX username and password.

### Step 3: Install Node.js
n8n requires a modern version of Node.js (v18.17+). The default version in Ubuntu is often too old, so we will install version 20 directly from NodeSource.

Run the following commands inside your **Ubuntu terminal**:

1.  **Update your package list:**
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2.  **Download the setup script for Node.js v20:**
    ```bash
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    ```

3.  **Install Node.js:**
    ```bash
    sudo apt-get install -y nodejs
    ```

4.  **Verify the installation:**
    ```bash
    node -v
    ```
    *Output should look like `v20.x.x`.*

5.  **Find your local windows IP in WSL**
    You will need the local windows IP in WSL. Run the following command:
    ``` bash
    ip route show | grep default
    ```
    * You will see output like: default via 172.25.0.1 dev eth0.
    * Copy or note that IP address (e.g., 172.25.0.1). This is your Windows host. You will need it for n8n soon.

### Step 3: Install and Start n8n
Now we will install n8n globally using the Node Package Manager (npm).

1.  **Install n8n:**
    ```bash
    sudo npm install n8n -g
    ```

2.  **Start the application:**
    ```bash
    n8n
    ```

3.  **Access the Interface:**
    * Look at the terminal output. It should say *"Editor is now accessible via..."*
    * Open your web browser in Windows and go to: `http://localhost:5678`
    * Follow the setup wizard to create your local owner account.

---

## 🤖 Part 2: Building the AI Workflow

**Scenario:** A local feedback analyzer that classifies customer emails without sending data to the cloud.

### Step 1: Create a New Workflow
1.  Once logged into n8n, look at the top right corner.
2.  Click the button **"Add workflow"**.
3.  You should now see a blank canvas with a grid background.

### Step 2: The Trigger & Data
We will simulate an incoming email.

1.  Click **"Add first step"**.
2.  Search for **"Manually trigger"**.
3.  Add the next node: Search for **"Edit Fields"**.
    * **Name:** `input_text`
    * **Value:** *"I am extremely disappointed. The package arrived late and it was broken. I want a refund immediately!"*

### Step 3: The AI Agent (The Brain)
Here we connect to Ollama.

1.  Add a new node after "Edit Fields". Search for **"AI Agent"**.
2.  **Configure the Agent:**
    * **Model Input:** Click the `+`-square below "Chat Model".
    * Search for **"Ollama Chat Model"**.
    * **Credentials:**
        * Create a new credential.
        * **Base URL:** `http://172.25.0.1:11434`
         (Replace 127.25.0.1 with your IP from before)
    * **Model Name:** Type the exact name of the model you pulled (e.g., `llama3.2`).
    * Click **"Add Option"**.
    * Select **"Format"**.
    * Change the dropdown to **"JSON"**.
    * *Explanation: This sends `format='json'` to the API, forcing the model to output valid JSON.*

3.  **Prompt (System Message):**
    * Change the mode from "Input Field" to **"Define below"** (or toggle "Expression" on).
    * **Paste this text:**
        > You are a support assistant. Analyze the email.
        > Customer Email: {{ $json.input_text }}
        > 
        > You must return a JSON object with exactly these two keys:
        > {
        >    "sentiment": "POSITIVE or NEGATIVE",
        >    "draft_reply": "A short, polite response string"
        > }

    * **Insert the Variable:**
      * Look at the "Input" panel on the left side of the window.
      * Find the `input_text` field coming from the previous node.
      * You can **Drag and drop** `input_text` into the prompt box, right after "Customer Email:".
      * *It should look like: `Customer Email: {{ $json.input_text }}`

### Step 4: Add Edit Fields Node
1.  Add an **"Edit Fields"** node after the AI Agent.
2.  **Mode:**
    * Manual Mapping.
3.  * **Add field for semtiment:**
      * CLick Add Field. Keep it as a String and name it "sentiment"
      * We extract the value from the AI-reply and use it as the value: {{ $json.output.parseJson().sentiment }}
    * **Add field for draft_reply:**
      * Add another field. Keep it as String and name it "draft_reply"
      * We extract the value from the AI-reply and  use it as the value: {{ $json.output.parseJson().draft_reply }}

### Step 5: The Logic (Branching)
1.  Add an **"If"** node after the Edit Fields-node.
2.  **Condition:**
    * **Value 1:** Drag `sentiment` from the Output Data panel.
    * **Operation:** Equal to
    * **Value 2:** `NEGATIVE`
3.  **Outputs:**
    * Connect a dummy node (e.g., "No Operation") to **True** (Label it: "Escalate").
    * Connect another dummy node to **False** (Label it: "Auto-reply").

---

## 🧪 Part 3: Testing & Troubleshooting

### Test the Flow
1.  Click **"Execute Workflow"**.
2.  The AI Agent might take a few seconds (running locally depends on your GPU/CPU).
3.  Check if it correctly categorized the angry text as "NEGATIVE".
4.  Try other email text to see if the agent categorizes them as you would.

### 💡 Troubleshooting WSL
* **"command not found: npm"**: You likely skipped the `source` command or didn't restart the terminal. Try closing and reopening Ubuntu.
* **n8n crashes**: Ensure you installed Node version 18 or 20. Run `node -v` to check.

* **Localhost not working**: Rarely, WSL doesn't map to localhost correctly. In PowerShell, run `wsl hostname -I` to get the IP, and try accessing `http://<YOUR-WSL-IP>:5678`.
---
















