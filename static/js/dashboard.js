// static/js/dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // --- Логика лайков/дизлайков ---
    function setButtonState(articleId, rating) {
        const articleDiv = document.querySelector(`.article-card[data-id="${articleId}"]`);
        if (!articleDiv) return;
        const likeBtn = articleDiv.querySelector('.btn-like');
        const dislikeBtn = articleDiv.querySelector('.btn-dislike');
        if (rating === 1) {
            likeBtn.classList.add('active');
            dislikeBtn.classList.remove('active');
        } else if (rating === 0) {
            dislikeBtn.classList.add('active');
            likeBtn.classList.remove('active');
        } else {
            likeBtn.classList.remove('active');
            dislikeBtn.classList.remove('active');
        }
    }

    document.querySelectorAll('.btn-like, .btn-dislike').forEach(btn => {
        btn.addEventListener('click', function() {
            const articleDiv = this.closest('.article-card');
            const articleId = articleDiv.dataset.id;
            const clickedRating = parseInt(this.dataset.rating);
            const currentRating = this.classList.contains('active') ? clickedRating : null;
            let newRating;
            if (currentRating !== null) {
                newRating = null;
            } else {
                newRating = clickedRating;
            }
            fetch(`/rate/${articleId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rating: newRating})
            }).then(res => res.json()).then(data => {
                if (data.status === 'ok') {
                    setButtonState(articleId, newRating);
                }
            });
        });
    });

    // --- Модальное окно для деталей статьи ---
    const modal = new bootstrap.Modal(document.getElementById('articleModal'));
    document.querySelectorAll('.btn-details').forEach(btn => {
        btn.addEventListener('click', function() {
            const articleId = this.dataset.id;
            const modalBody = document.getElementById('modalBody');
            modalBody.innerHTML = '<p>Loading...</p>';
            fetch(`/article/${articleId}`)
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        modalBody.innerHTML = `<p class="text-danger">${escapeHtml(data.error)}</p>`;
                    } else {
                        modalBody.innerHTML = `
                            <h6>${escapeHtml(data.title)}</h6>
                            <p class="text-muted">${escapeHtml(data.source)} | ${escapeHtml(data.date)}</p>
                            <hr>
                            <p>${escapeHtml(data.text) || 'No full text available.'}</p>
                        `;
                        document.getElementById('modalOriginalLink').href = data.url;
                    }
                })
                .catch(err => {
                    modalBody.innerHTML = '<p class="text-danger">Failed to load article details.</p>';
                });
            modal.show();
        });
    });

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    // --- Экспорт BibTeX ---
    document.getElementById('export-bibtex')?.addEventListener('click', function() {
        window.location.href = '/export/bibtex';
    });

    // --- Подсказки и команды (Agentic UI) ---
    function showContextualTip() {
        fetch('/api/user_context')
            .then(res => res.json())
            .then(data => {
                const insightDiv = document.getElementById('quick-insight');
                const insightText = document.getElementById('insight-text');
                if (data.show_tip) {
                    insightText.textContent = data.message;
                    insightDiv.style.display = 'block';
                } else {
                    insightDiv.style.display = 'none';
                }
            });
    }

    window.dismissInsight = function() {
        document.getElementById('quick-insight').style.display = 'none';
    };

    // --- Поиск и обновление дайджеста ---
    function showMessage(text, duration = 3000) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'search-message';
        msgDiv.textContent = text;
        document.body.appendChild(msgDiv);
        setTimeout(() => msgDiv.remove(), duration);
    }

    function updateDigest(query) {
        const skeleton = document.getElementById('skeleton-loader');
        const articlesList = document.getElementById('articles-list');
        // Показываем скелетон
        if (skeleton) skeleton.style.display = 'block';
        if (articlesList) articlesList.style.display = 'none';

        fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query})
        })
        .then(res => res.json())
        .then(data => {
            if (data.html) {
                // Заменяем содержимое articles-list
                articlesList.innerHTML = data.html;
                // Скрываем скелетон, показываем список
                skeleton.style.display = 'none';
                articlesList.style.display = 'block';
                // Повторно привязываем обработчики кнопок (лайки, детали)
                attachHandlers();
                // Показываем сообщение
                showMessage(`Digest updated: showing results for "${query}"`);
            } else if (data.error) {
                showMessage(data.error, 4000);
                skeleton.style.display = 'none';
                articlesList.style.display = 'block';
            }
        })
        .catch(err => {
            console.error('Search error:', err);
            showMessage('Failed to update digest. Please try again.', 4000);
            skeleton.style.display = 'none';
            articlesList.style.display = 'block';
        });
    }

    function attachHandlers() {
        // Лайки/дизлайки
        document.querySelectorAll('.btn-like, .btn-dislike').forEach(btn => {
            btn.removeEventListener('click', handleLikeDislike);
            btn.addEventListener('click', handleLikeDislike);
        });
        // Детали
        document.querySelectorAll('.btn-details').forEach(btn => {
            btn.removeEventListener('click', handleDetails);
            btn.addEventListener('click', handleDetails);
        });
    }

    function handleLikeDislike(e) {
        const btn = e.currentTarget;
        const articleDiv = btn.closest('.article-card');
        const articleId = articleDiv.dataset.id;
        const clickedRating = parseInt(btn.dataset.rating);
        const currentRating = btn.classList.contains('active') ? clickedRating : null;
        let newRating;
        if (currentRating !== null) {
            newRating = null;
        } else {
            newRating = clickedRating;
        }
        fetch(`/rate/${articleId}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({rating: newRating})
        }).then(res => res.json()).then(data => {
            if (data.status === 'ok') {
                setButtonState(articleId, newRating);
            }
        });
    }

    function handleDetails(e) {
        const btn = e.currentTarget;
        const articleId = btn.dataset.id;
        const modalBody = document.getElementById('modalBody');
        modalBody.innerHTML = '<p>Loading...</p>';
        fetch(`/article/${articleId}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    modalBody.innerHTML = `<p class="text-danger">${escapeHtml(data.error)}</p>`;
                } else {
                    modalBody.innerHTML = `
                        <h6>${escapeHtml(data.title)}</h6>
                        <p class="text-muted">${escapeHtml(data.source)} | ${escapeHtml(data.date)}</p>
                        <hr>
                        <p>${escapeHtml(data.text) || 'No full text available.'}</p>
                    `;
                    document.getElementById('modalOriginalLink').href = data.url;
                }
            })
            .catch(err => {
                modalBody.innerHTML = '<p class="text-danger">Failed to load article details.</p>';
            });
        modal.show();
    }

    function processCommand() {
        const input = document.getElementById('command-input').value.trim();
        if (!input) return;
        updateDigest(input);
        document.getElementById('command-input').value = '';
    }

    document.getElementById('execute-command')?.addEventListener('click', processCommand);
    document.getElementById('command-input')?.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') processCommand();
    });

    // Показываем подсказку при загрузке
    showContextualTip();

    // Имитация загрузки: скрываем скелетон, показываем список (если есть)
    setTimeout(function() {
        const skeleton = document.getElementById('skeleton-loader');
        const articlesList = document.getElementById('articles-list');
        if (skeleton && articlesList) {
            skeleton.style.display = 'none';
            articlesList.style.display = 'block';
        }
        attachHandlers();
    }, 500);
});