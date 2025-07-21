$(document).ready(function () {

    function detectThemeClass() {
        const theme = document.documentElement.getAttribute("data-bs-theme") || "light";
        return theme === "dark" ? "select2-dark" : "select2-light";
    }

    function applyThemeToDropdown() {
        const theme = detectThemeClass();
        $('.select2-dropdown').removeClass('select2-dark select2-light').addClass(theme);
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

    function initAjaxSelect2($select) {
    if ($select.hasClass('select2-hidden-accessible')) {
        $select.select2('destroy');
    }

    const $modal = $select.closest('.modal');

    $select.select2({
        dropdownParent: $modal,
        width: '100%',
        tags: true,
        language: {
            inputTooShort: function () { return "Tapez pour rechercher…"; },
            noResults: function () { return "Aucun résultat trouvé."; }
        },
        placeholder: "Tapez pour rechercher ou créer…",
        minimumInputLength: 1,
        ajax: {
            url: '/api/groups/search',
            dataType: 'json',
            delay: 250,
            data: function (params) {
                return { q: params.term };
            },
            processResults: function (data) {
                if (!data.results) return { results: [] };
                // Les clés attendues par Select2 sont `id` et `text`
                return {
                    results: data.results.map(item => ({
                        id: item.id,
                        text: item.text
                    }))
                };
            },
            cache: true
        }
    }).on('select2:open', applyThemeToDropdown);
}

    initSelect2();

    $('.select2-ajax').each(function () {
        initAjaxSelect2($(this));
    });

    $(document).on('shown.bs.modal', '.modal', function () {
        $(this).find('.select2-ajax').each(function () {
            initAjaxSelect2($(this));
        });
    });

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
            .done(() => {
                const currentPage = new URLSearchParams(window.location.search).get('page') || '1';
                window.location.href = `/print_history?page=${currentPage}&focus_print_id=${printId}`;
            })
            .fail(() => alert('Erreur lors de l’ajout du tag.'));
    });

    $('.remove-tag').on('click', function () {
        const btn = $(this);
        const printId = btn.data('print-id');
        const tag = btn.data('tag');

        $.post(`/history/${printId}/tags/remove`, { tag })
            .done(() => {
                const currentPage = new URLSearchParams(window.location.search).get('page') || '1';
                window.location.href = `/print_history?page=${currentPage}&focus_print_id=${printId}`;
            })
            .fail(() => alert('Erreur lors de la suppression du tag.'));
    });

});

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

    $colorSelect.select2({
        width: '100%',
        templateResult: formatColorOption,
        templateSelection: formatColorOption,
        escapeMarkup: m => m
    });

    applyColorTags();
    $colorSelect.on('select2:select select2:unselect', () => applyColorTags());
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
        const val = $(this).attr('title');
        const enName = Object.keys(COLOR_NAME_MAP).find(k => COLOR_NAME_MAP[k] === val) || val;
        const hex = getFamilyHex(enName);
        const label = val;

        $(this).html(`
            <span class="select2-selection__choice__remove" role="presentation">×</span>
            <span style="
                display: inline-block;
                width: 12px;
                height: 12px;
                margin: 0 4px;
                background: ${hex};
                border: 1px solid #ccc;
                border-radius: 2px;
                vertical-align: middle;
            "></span>
            <span>${label}</span>
        `);
    });
}
