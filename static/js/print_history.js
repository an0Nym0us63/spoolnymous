function detectThemeClass() {
    const theme = document.documentElement.getAttribute("data-bs-theme") || "light";
    return theme === "dark" ? "select2-dark" : "select2-light";
}

$(document).ready(function () {
    function applyThemeToDropdown() {
        const theme = document.documentElement.getAttribute("data-bs-theme") || "light";
        const classToAdd = theme === "dark" ? "select2-dark" : "select2-light";
        $('.select2-dropdown').removeClass('select2-dark select2-light').addClass(classToAdd);
    }

    function initSelect2() {
    $('.select2').each(function () {
        if ($(this).hasClass('select2-hidden-accessible')) {
            $(this).select2('destroy');
        }
        $(this).select2({ width: '100%' }).on('select2:open', applyThemeToDropdown);
    });
	 enhanceColorSelect();
applyColorTags();
}

    initSelect2();

    $('#filtersCollapse').on('shown.bs.collapse', function () {
        initSelect2();
    });
	 $('.add-tag-btn').on('click', function () {
    const btn = $(this);
    const printId = btn.data('print-id');
    const input = btn.siblings('.add-tag-input');
    const tag = input.val().trim();

    if (!tag) return;

    $.post(`/history/${printId}/tags/add`, { tag })
      .done(() => location.reload())
      .fail(() => alert('Erreur lors de l’ajout du tag.'));
  });

  $('.remove-tag').on('click', function () {
    const btn = $(this);
    const printId = btn.data('print-id');
    const tag = btn.data('tag');

    $.post(`/history/${printId}/tags/remove`, { tag })
      .done(() => location.reload())
      .fail(() => alert('Erreur lors de la suppression du tag.'));
  });
});

function confirmReajust(printId) {
    askRestockRatioPerFilament(printId, false);
}

function confirmDelete(printId) {
    askRestockRatioPerFilament(printId, true);
}

function askRestockRatioPerFilament(printId, isDelete) {
    fetch(`/history/${printId}/filaments`)
    .then(resp => resp.json())
    .then(filaments => {
        const formHtml = filaments.map(f => `
            <div style="display:flex;align-items:center;margin-bottom:5px;gap:5px">
                <div style="width:15px;height:15px;background:${f.color};border:1px solid #ccc"></div>
                <span style="flex:1">${f.name}</span>
                <input type="number" min="0" max="100" value="100" id="ratio_${f.spool_id}" style="width:60px"> %
            </div>
        `).join("");

        Swal.fire({
            title: isDelete ? "Supprimer + Réajuster" : "Réajuster uniquement",
            html: formHtml,
            showCancelButton: true,
            confirmButtonText: "Valider",
            cancelButtonText: "Annuler",
            preConfirm: () => {
                const ratios = {};
                filaments.forEach(f => {
                    const val = parseInt(document.getElementById(`ratio_${f.spool_id}`).value) || 0;
                    ratios[f.spool_id] = val;
                });
                return ratios;
            }
        }).then(result => {
            if (result.isConfirmed) {
                const url = isDelete
                    ? `/history/delete/${printId}`
                    : `/history/reajust/${printId}`;

                fetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ restock: true, ratios: result.value })
                })
                .then(resp => resp.json())
                .then(data => {
                    if (data.status && data.status.toLowerCase() === "ok") {
                        location.reload(); // Force le refresh pour éviter des états désynchronisés
                    } else {
                        Swal.fire("Erreur", data.error || "Impossible d'effectuer l'opération", "error");
                    }
                });
            }
        });
    });
}


function addTag(printId, tag) {
    fetch(`/history/${printId}/tags/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tag })
    }).then(() => location.reload());
}

function removeTag(printId, tag) {
    fetch(`/history/${printId}/tags/remove`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tag })
    }).then(() => location.reload());
}

const COLOR_NAME_MAP = {
    "Black": "Noir",
    "White": "Blanc",
    "Grey": "Gris",
    "Red": "Rouge",
    "Dark Red": "Rouge foncé",
    "Pink": "Rose",
    "Magenta": "Magenta",
    "Brown": "Marron",
    "Yellow": "Jaune",
    "Gold": "Doré",
    "Orange": "Orange",
    "Green": "Vert",
    "Dark Green": "Vert foncé",
    "Lime": "Vert fluo",
    "Teal": "Turquoise",
    "Blue": "Bleu",
    "Navy": "Bleu marine",
    "Cyan": "Cyan",
    "Lavender": "Lavande",
    "Purple": "Violet",
    "Dark Purple": "Violet foncé"
};

function enhanceColorSelect() {
    const $colorSelect = $('select[name="color"]');

    // Collect current options and replace with enhanced sorted ones
    const options = $colorSelect.find('option').map(function () {
        const value = $(this).val();
        const selected = $(this).is(':selected');
        const label = COLOR_NAME_MAP[value] || value;
        return { value, label, selected };
    }).get();

    options.sort((a, b) => a.label.localeCompare(b.label, 'fr'));

    $colorSelect.empty();

    for (const opt of options) {
        const $opt = $('<option>')
            .val(opt.value)
            .text(opt.label)
            .attr('data-color', opt.value)
            .prop('selected', opt.selected);
        $colorSelect.append($opt);
    }

    // Enhance with Select2 + template
    $colorSelect.select2({
    width: '100%',
    templateResult: formatColorOption,
    templateSelection: formatColorOption,
    escapeMarkup: function(m) { return m; }
});
}

function formatColorOption(state) {
    if (!state.id) return state.text;
    const colorHex = getFamilyHex(state.id);
    const label = state.text;
    const $el = $(`
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:14px;height:14px;border-radius:3px;background:${colorHex};border:1px solid #ccc"></span>
            <span>${label}</span>
        </div>
    `);
    return $el;
}

function getFamilyHex(name) {
    const map = {
        "Black": "#000000",
        "White": "#FFFFFF",
        "Grey": "#A0A0A0",
        "Red": "#DC143C",
        "Dark Red": "#8B0000",
        "Pink": "#FFB6C1",
        "Magenta": "#FF00FF",
        "Brown": "#964B00",
        "Yellow": "#FFDC00",
        "Gold": "#D4AF37",
        "Orange": "#FF8C00",
        "Green": "#50C878",
        "Dark Green": "#006400",
        "Lime": "#BFFF00",
        "Teal": "#008080",
        "Blue": "#6496FF",
        "Navy": "#000080",
        "Cyan": "#00FFFF",
        "Lavender": "#E6E6FA",
        "Purple": "#A020F0",
        "Dark Purple": "#5A3C78"
    };
    return map[name] || "#CCCCCC";
}

function applyColorTags() {
    $('.select2-selection__choice').each(function () {
        const val = $(this).attr('title'); // valeur réelle (en anglais)
        const hex = getFamilyHex(val);
        $(this).css({
            display: 'flex',
            'align-items': 'center',
            'gap': '4px'
        });
        $(this).prepend(`<span style="
            display:inline-block;
            width:10px;
            height:10px;
            background:${hex};
            border:1px solid #ccc;
            border-radius:2px;
        "></span>`);
    });
}