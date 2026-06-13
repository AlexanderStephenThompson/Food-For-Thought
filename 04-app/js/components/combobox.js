// Attach the ARIA combobox pattern to an input and listbox pair.
// Caller supplies getMatches(query) -> [{ id, name, meta }] and an
// onSelect(match) callback. Handles open/close, keyboard navigation,
// and the aria-* state the pattern requires.

const KEY_DOWN = "ArrowDown";
const KEY_UP = "ArrowUp";
const KEY_ENTER = "Enter";
const KEY_ESCAPE = "Escape";

export function attachCombobox({ input, listbox, getMatches, onSelect }) {
  let matches = [];
  let activeIndex = -1;

  function closeListbox() {
    listbox.hidden = true;
    listbox.replaceChildren();
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    activeIndex = -1;
  }

  function renderOption(match, position) {
    const option = document.createElement("li");
    option.className = "combobox__option";
    option.id = `${listbox.id}-option-${position}`;
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");

    const name = document.createElement("span");
    name.className = "combobox__option-name";
    name.textContent = match.name;
    option.append(name);

    if (match.meta) {
      const meta = document.createElement("span");
      meta.className = "combobox__option-meta";
      meta.textContent = match.meta;
      option.append(meta);
    }

    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      choose(position);
    });
    return option;
  }

  function openWith(query) {
    matches = getMatches(query);
    if (matches.length === 0) {
      closeListbox();
      return;
    }
    listbox.replaceChildren(...matches.map(renderOption));
    listbox.hidden = false;
    input.setAttribute("aria-expanded", "true");
    activeIndex = -1;
  }

  function highlight(nextIndex) {
    const options = [...listbox.children];
    options.forEach((option, position) => {
      option.setAttribute("aria-selected", position === nextIndex ? "true" : "false");
    });
    activeIndex = nextIndex;
    if (nextIndex >= 0) {
      input.setAttribute("aria-activedescendant", options[nextIndex].id);
      options[nextIndex].scrollIntoView({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function choose(position) {
    const match = matches[position];
    if (!match) {
      return;
    }
    onSelect(match);
    input.value = "";
    closeListbox();
    input.focus();
  }

  input.addEventListener("input", () => openWith(input.value));

  input.addEventListener("keydown", (event) => {
    if (listbox.hidden && event.key === KEY_DOWN) {
      openWith(input.value);
      return;
    }
    if (listbox.hidden) {
      return;
    }
    if (event.key === KEY_DOWN) {
      event.preventDefault();
      highlight((activeIndex + 1) % matches.length);
    } else if (event.key === KEY_UP) {
      event.preventDefault();
      highlight((activeIndex - 1 + matches.length) % matches.length);
    } else if (event.key === KEY_ENTER && activeIndex >= 0) {
      event.preventDefault();
      choose(activeIndex);
    } else if (event.key === KEY_ESCAPE) {
      closeListbox();
    }
  });

  input.addEventListener("blur", () => {
    window.setTimeout(closeListbox, 0);
  });
}
