const input    = document.getElementById("movieInput");
const btn      = document.getElementById("searchBtn");
const error    = document.getElementById("error");
const results  = document.getElementById("results");
const heading  = document.getElementById("resultsHeading");
const cards    = document.getElementById("cards");
const loading  = document.getElementById("loading");
const dropdown = document.getElementById("dropdown");

// --- Autocomplete ---

let allMovies = [];
let activeIndex = -1;

// Fetch movie list once on load
fetch("/api/movies")
  .then(r => r.json())
  .then(data => { allMovies = data.movies || []; })
  .catch(() => { console.warn("Could not load movie list for autocomplete."); });

input.addEventListener("input", () => {
  const query = input.value.trim();
  if (query.length < 2) { closeDropdown(); return; }

  const matches = allMovies
    .filter(t => t.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 8);

  if (matches.length === 0) { closeDropdown(); return; }

  activeIndex = -1;
  dropdown.innerHTML = "";
  matches.forEach((title, i) => {
    const li = document.createElement("li");
    li.textContent = title;
    li.setAttribute("role", "option");
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // prevent blur before click fires
      selectTitle(title);
    });
    dropdown.appendChild(li);
  });
  show(dropdown);
});

input.addEventListener("keydown", (e) => {
  const items = dropdown.querySelectorAll("li");

  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeIndex = Math.min(activeIndex + 1, items.length - 1);
    updateActive(items);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeIndex = Math.max(activeIndex - 1, -1);
    updateActive(items);
  } else if (e.key === "Enter") {
    if (activeIndex >= 0 && items[activeIndex]) {
      selectTitle(items[activeIndex].textContent);
    } else {
      closeDropdown();
      getRecommendations();
    }
  } else if (e.key === "Escape") {
    closeDropdown();
  }
});

input.addEventListener("blur", () => {
  // Small delay so mousedown on a list item fires first
  setTimeout(closeDropdown, 150);
});

function updateActive(items) {
  items.forEach((li, i) => li.classList.toggle("active", i === activeIndex));
  if (activeIndex >= 0) items[activeIndex].scrollIntoView({ block: "nearest" });
}

function selectTitle(title) {
  input.value = title;
  closeDropdown();
  getRecommendations();
}

function closeDropdown() {
  hide(dropdown);
  activeIndex = -1;
}

// --- Recommendations ---

async function getRecommendations() {
  const title = input.value.trim();
  if (!title) return;

  hide(error);
  hide(results);
  show(loading);
  btn.disabled = true;

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie_title: title, top_n: 5 }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong. Please try again.");
      return;
    }

    renderResults(title, data.recommendations);
  } catch {
    showError("Could not reach the server. Please try again.");
  } finally {
    hide(loading);
    btn.disabled = false;
  }
}

function renderResults(title, recs) {
  cards.innerHTML = "";

  if (!recs || recs.length === 0) {
    showError(`No recommendations found for "${title}".`);
    return;
  }

  heading.textContent = `Books recommended for "${title}"`;

  recs.forEach((rec) => {
    const pct = Math.round(rec.similarity * 100);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-info">
        <div class="card-title">${escapeHtml(rec.book_title)}</div>
        <div class="card-author">by ${escapeHtml(rec.book_author)}</div>
      </div>
      <div class="card-score">${pct}% match</div>
    `;
    cards.appendChild(card);
  });

  show(results);
}

function showError(msg) {
  error.textContent = msg;
  show(error);
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
