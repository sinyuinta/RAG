document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatArea = document.getElementById('chat-area');
    const sendBtn = document.getElementById('send-btn');
    const welcomeMessage = document.querySelector('.welcome-message');

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';

        // Enable/disable send button
        if (this.value.trim().length > 0) {
            sendBtn.removeAttribute('disabled');
        } else {
            sendBtn.setAttribute('disabled', 'true');
        }
    });

    // Handle Enter key (Shift+Enter for new line)
    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim().length > 0) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Handle suggestion buttons
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('suggestion-btn')) {
            e.preventDefault();
            e.stopPropagation();
            userInput.value = e.target.textContent;
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
            sendBtn.removeAttribute('disabled');
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Handle submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // Hide welcome message if visible
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }

        // Add user message
        addMessage(query, 'user');

        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = '48px'; // Fixed initial height reset
        sendBtn.setAttribute('disabled', 'true');

        // Show loading state
        const loadingId = addLoadingIndicator();
        scrollToBottom(); // Ensure we scroll when loading appears

        // Get selected mode
        const mode = document.querySelector('input[name="search-mode"]:checked').value;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    mode: mode
                })
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();

            // Remove loading indicator
            removeMessage(loadingId);

            // Add bot message
            addBotMessage(data);

        } catch (error) {
            console.error('Error:', error);
            removeMessage(loadingId);
            addMessage('申し訳ありません。エラーが発生しました。もう一度お試しください。', 'bot');
        }
    });

    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (role === 'bot') {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(contentDiv);
        chatArea.appendChild(messageDiv);
        scrollToBottom();
    }

    function addBotMessage(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Main answer parsed with markdown
        let htmlContent = marked.parse(data.answer);

        // Standardized match level badge parsing with Icons (Supports H2 and H3)
        htmlContent = htmlContent.replace(/<(h[23])>(.*?) (高|中|低)<\/\1>/g, (match, tag, title, level) => {
            let levelClass = level === '高' ? 'match-high' : (level === '中' ? 'match-medium' : 'match-low');
            let icon = '';
            if (level === '高') {
                icon = '<svg class="match-icon" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/></svg>';
            } else if (level === '中') {
                icon = '<svg class="match-icon" viewBox="0 0 24 24"><path d="M12 2C6.47 2 2 6.47 2 12C2 17.53 6.47 22 12 22C17.53 22 22 17.53 22 12C22 6.47 17.53 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z"/></svg>';
            } else {
                icon = '<svg class="match-icon" viewBox="0 0 24 24"><path d="M22 12C22 17.52 17.52 22 12 22C6.48 22 2 17.52 2 12C2 6.48 6.48 2 12 2C17.52 2 22 6.48 22 12ZM7 13H17V11H7V13Z"/></svg>';
            }

            // Apply school specific colors based on title content
            let schoolClass = '';
            if (title.includes('アドラー')) schoolClass = 'school-adler';
            else if (title.includes('フロイト')) schoolClass = 'school-freud';
            else if (title.includes('ユング')) schoolClass = 'school-jung';
            else if (title.includes('認知')) schoolClass = 'school-neisser';

            return `<${tag} class="${schoolClass}">${title} <span class="match-tag ${levelClass}">${icon}${level}</span></${tag}>`;
        });

        // Add sources if available
        if (data.sources && data.sources.length > 0) {
            let sourcesHtml = `
                <div class="sources-section">
                    <div class="sources-title">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="currentColor"/>
                        </svg>
                        内部ライブラリー出典
                    </div>
            `;

            // Show the top 5 most relevant sources
            data.sources.slice(0, 5).forEach((source, index) => {
                const meta = source.metadata;
                let fileName = meta.source_file || '不明';
                // Ensure .md extension if missing to avoid 404
                if (fileName !== '不明' && !fileName.endsWith('.md')) {
                    fileName += '.md';
                }

                sourcesHtml += `
                    <div class="source-item">
                        <div class="source-meta"><span class="source-index">【参考文献${index + 1}】</span> ${meta.ancestor} (${meta.family_line})</div>
                        <div class="source-link">
                             <svg width="12" height="12" viewBox="0 0 24 24" fill="none" style="display:inline; vertical-align:middle; margin-right:4px;">
                                <path d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                             </svg>
                             <a href="/docs/${fileName}" target="_blank">${meta.section_title}</a>
                        </div>
                    </div>
                `;
            });

            sourcesHtml += '</div>';
            htmlContent += sourcesHtml;
        }

        contentDiv.innerHTML = htmlContent;
        messageDiv.appendChild(contentDiv);
        chatArea.appendChild(messageDiv);
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.id = id;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        `;

        messageDiv.appendChild(contentDiv);
        chatArea.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeMessage(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
});
