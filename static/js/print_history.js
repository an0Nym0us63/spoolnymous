$(document).ready(function () {

    function detectThemeClass() {
        const theme = document.documentElement.getAttribute("data-bs-theme") || "light";
        return theme === "dark" ? "select2-dark" : "select2-light";
    }
	
	function formatStatusOption(state) {
    if (!state.id) return state.text;

    const colorMap = {
        "SUCCESS": "#198754",
        "TO_REDO": "#ffc107",
        "PARTIAL": "#fd7e14",
        "FAILED": "#dc3545",
        "IN_PROGRESS": "#0dcaf0"
    };
    const id = state.id.toUpperCase(); // Sécurité
    const color = colorMap[id] || "#6c757d";

    return $(`
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:14px;height:14px;border-radius:3px;background:${color};border:1px solid #ccc;"></span>
            <span>${state.text}</span>
        </div>
    `);
}

    function applyThemeToDropdown() {
        const theme = detectThemeClass();
        $('.select2-dropdown').removeClass('select2-dark select2-light').addClass(theme);
    }

    function getSwalThemeClasses() {
        const theme = document.documentElement.getAttribute("data-bs-theme") || "light";
        if (theme === "dark") {
            return {
                popup: 'bg-dark text-light',
                title: 'text-light',
                htmlContainer: 'text-light',
                confirmButton: 'btn btn-primary',
                cancelButton: 'btn btn-secondary'
            };
        } else {
            return {
                confirmButton: 'btn btn-primary',
                cancelButton: 'btn btn-secondary'
            };
        }
    }

    function initSelect2() {
    $('.select2').each(function () {
        const $select = $(this);
        const name = $select.attr('name');

        // Supprime proprement si déjà instancié
        if ($select.hasClass('select2-hidden-accessible')) {
            $select.select2('destroy');
        }

        const config = {
            width: '100%',
            placeholder: $select.data('placeholder') || '',
            allowClear: !$select.prop('multiple')
        };

        if (name === 'status') {
            Object.assign(config, {
                templateResult: formatStatusOption,
                templateSelection: formatStatusOption,
                escapeMarkup: m => m
            });
        }

        if (name === 'color') {
            Object.assign(config, {
                templateResult: formatColorOption,
                templateSelection: formatColorOption,
                escapeMarkup: m => m
            });
        }

        if (name === 'group_id_or_name') {
            Object.assign(config, {
                tags: true,
                dropdownParent: $select.closest('.modal'),
                placeholder: "Tapez pour rechercher ou créer…",
                minimumInputLength: 1,
                language: {
                    inputTooShort: () => "Commencez à taper pour chercher ou créer…"
                },
                ajax: {
                    url: '/api/groups/search',
                    dataType: 'json',
                    delay: 250,
                    data: params => ({ q: params.term }),
                    processResults: data => ({
                        results: (data.results || []).map(g => ({ id: g.id, text: g.text }))
                    }),
                    cache: true
                }
            });
        }

        $select.select2(config).on('select2:open', applyThemeToDropdown);
    });

    // Applique les pastilles aux statuts déjà sélectionnés
    $('select[name="status"]').each(function () {
        const $select = $(this);
		 // ⚠️ Check : si select2 n'est pas encore actif, on saute
    if (!$select.hasClass('select2-hidden-accessible')) return;
        const data = $select.select2('data');
        const colorMap = {
            "SUCCESS": "#198754",
            "TO_REDO": "#ffc107",
            "PARTIAL": "#fd7e14",
            "FAILED": "#dc3545",
            "IN_PROGRESS": "#0dcaf0"
        };

        $select.next('.select2-container').find('.select2-selection__choice').each(function (i) {
            const state = data[i];
            if (!state) return;
            const color = colorMap[state.id?.toUpperCase()] || "#6c757d";

            $(this).html(`
                <span class="select2-selection__choice__remove" role="presentation">×</span>
                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    margin: 0 4px;
                    background: ${color};
                    border: 1px solid #ccc;
                    border-radius: 2px;
                    vertical-align: middle;"></span>
                <span>${state.text}</span>
            `);
        });
    });

    enhanceColorSelect();  // pour l’ordre des familles
    applyColorTags();      // pour les pastilles couleurs
}


    function initAjaxSelect2($select) {
        if ($select.hasClass('select2-hidden-accessible')) {
            $select.select2('destroy');
        }

        $select.select2({
            width: '100%',
            tags: true,
            dropdownParent: $select.closest('.modal'),
            placeholder: "Tapez pour rechercher ou créer…",
            minimumInputLength: 1,
            language: {
                inputTooShort: () => "Commencez à taper pour chercher ou créer…"
            },
            ajax: {
                url: '/api/groups/search',
                dataType: 'json',
                delay: 250,
                data: params => ({ q: params.term }),
                processResults: data => ({
                    results: (data.results || []).map(g => ({ id: g.id, text: g.text }))
                }),
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

    window.confirmReajust = function (printId) {
        askRestockRatioPerFilament(printId, false);
    };

    window.confirmDelete = function (printId) {
        askRestockRatioPerFilament(printId, true);
    };

    function askRestockRatioPerFilament(printId, isDelete) {
        fetch(`/history/${printId}/filaments`)
            .then(resp => resp.json())
            .then(filaments => {
                const formHtml = `
                    <div id="ratiosForm">
                        ${filaments.map(f => `
                            <div style="display:flex;align-items:center;margin-bottom:5px;gap:5px">
                                <div style="width:15px;height:15px;background:${f.color};border:1px solid #ccc"></div>
                                <span style="flex:1">${f.name}</span>
                                <input type="number" min="0" max="100" value="${isDelete ? 100 : 0}" id="ratio_${f.spool_id}" style="width:60px"> %
                            </div>
                        `).join("")}
                    </div>
                    <div class="d-flex justify-content-around mt-2">
                        ${[0, 25, 50, 75, 100].map(val => `
                            <button type="button" class="btn btn-sm btn-outline-primary preset-btn" data-value="${val}">${val}%</button>
                        `).join("")}
                    </div>
                `;

                Swal.fire({
                    title: isDelete ? "Supprimer + Réajuster" : "Réajuster uniquement",
                    html: formHtml,
                    showCancelButton: true,
                    confirmButtonText: "Valider",
                    cancelButtonText: "Annuler",
                    customClass: getSwalThemeClasses(),
                    didOpen: () => {
                        document.querySelectorAll('.preset-btn').forEach(btn => {
                            btn.addEventListener('click', () => {
                                const val = btn.getAttribute('data-value');
                                filaments.forEach(f => {
                                    document.getElementById(`ratio_${f.spool_id}`).value = val;
                                });
                            });
                        });
                    },
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

                        const currentPage = new URLSearchParams(window.location.search).get('page') || '1';

                        fetch(url, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ restock: true, ratios: result.value })
                        })
                            .then(resp => resp.json())
                            .then(data => {
                                if (data.status?.toLowerCase() === "ok") {
                                    window.location.href = `/print_history?page=${currentPage}&focus_print_id=${printId}`;
                                } else {
                                    Swal.fire("Erreur", data.error || "Échec de l’opération", "error");
                                }
                            });
                    }
                });
            });
    }
	window.confirmAdjustDuration = function (printId) {
    const hourInput = document.getElementById(`hours_${printId}`).value;
    const minInput = document.getElementById(`minutes_${printId}`).value;

    const hours = parseFloat(hourInput || "0");
    const minutes = parseFloat(minInput || "0");

    if ((isNaN(hours) && isNaN(minutes)) || (hours <= 0 && minutes <= 0)) {
        Swal.fire({
            title: "Durée vide",
            text: "Merci de saisir une durée dans au moins un des deux champs.",
            icon: "warning",
            confirmButtonText: "OK",
            customClass: getSwalThemeClasses()
        });
        return;
    }

    const totalMinutes = (isNaN(hours) ? 0 : hours * 60) + (isNaN(minutes) ? 0 : minutes);
    const hFinal = Math.floor(totalMinutes / 60);
    const mFinal = Math.round(totalMinutes % 60);

    const msg = `Confirmer l’ajustement de la durée à ${hFinal}h${mFinal > 0 ? ' ' + mFinal + 'min' : ''} ?`;

    Swal.fire({
        title: "Confirmer l’ajustement",
        text: msg,
        icon: "question",
        showCancelButton: true,
        confirmButtonText: "Oui",
        cancelButtonText: "Annuler",
        customClass: getSwalThemeClasses()
    }).then(result => {
        if (result.isConfirmed) {
            document.querySelector(`#adjustDurationModal_${printId} form`).submit();
        }
    });
};
});

// Fonctions couleurs
const COLOR_NAME_MAP = {
    "Black": "Noir", "White": "Blanc", "Grey": "Gris", "Red": "Rouge", "Dark Red": "Rouge foncé",
    "Pink": "Rose", "Magenta": "Magenta", "Brown": "Marron", "Yellow": "Jaune", "Gold": "Doré",
    "Orange": "Orange", "Green": "Vert", "Dark Green": "Vert foncé", "Lime": "Vert fluo", "Teal": "Turquoise",
    "Blue": "Bleu", "Navy": "Bleu marine", "Cyan": "Cyan", "Lavender": "Lavande",
    "Purple": "Violet", "Dark Purple": "Violet foncé"
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
        $('<option>')
            .val(opt.value)
            .text(opt.label)
            .attr('data-color', opt.value)
            .prop('selected', opt.selected)
            .appendTo($colorSelect);
    }
$colorSelect.select2({
    width: '100%',
    placeholder: "— Filtrer par famille de couleur —",
    allowClear: true,
    templateResult: formatColorOption,
    templateSelection: formatColorOption,
    escapeMarkup: m => m
});

    applyColorTags();
    $colorSelect.on('select2:select select2:unselect', applyColorTags);
}

function formatFilamentOption(option) {
    if (!option.id) return option.text;
    const color = $(option.element).data('color');
    const small = `style="font-size: 0.9rem; line-height: 1.2;"`;

    const swatch = color
        ? `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color};margin-right:6px;"></span>`
        : '';

    return $(`<span ${small}>${swatch}${option.text}</span>`);
}

function formatColorOption(state) {
    if (!state.id) return state.text;
    const colorHex = getFamilyHex(state.id);
    return $(`
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="width:14px;height:14px;border-radius:3px;background:${colorHex};border:1px solid #ccc"></span>
            <span>${state.text}</span>
        </div>
    `);
}

function getFamilyHex(name) {
    const map = {
        "Black": "#000000", "White": "#FFFFFF", "Grey": "#A0A0A0", "Red": "#DC143C",
        "Dark Red": "#8B0000", "Pink": "#FFB6C1", "Magenta": "#FF00FF", "Brown": "#964B00",
        "Yellow": "#FFDC00", "Gold": "#D4AF37", "Orange": "#FF8C00", "Green": "#50C878",
        "Dark Green": "#006400", "Lime": "#BFFF00", "Teal": "#008080", "Blue": "#6496FF",
        "Navy": "#000080", "Cyan": "#00FFFF", "Lavender": "#E6E6FA", "Purple": "#A020F0",
        "Dark Purple": "#5A3C78"
    };
    return map[name] || "#CCCCCC";
}

function applyColorTags() {
    $('.select2-selection__choice').each(function () {
        const val = $(this).attr('title');
        const enName = Object.keys(COLOR_NAME_MAP).find(k => COLOR_NAME_MAP[k] === val) || val;
        const hex = getFamilyHex(enName);
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
                vertical-align: middle;"></span>
            <span>${val}</span>
        `);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const accordions = document.querySelectorAll(".card-header[data-bs-toggle='collapse']");

    // Gestion du clic classique sur un header
    accordions.forEach(header => {
        header.addEventListener("click", () => {
            const targetSelector = header.getAttribute("data-bs-target");
            const target = document.querySelector(targetSelector);

            if (!target.classList.contains("show")) {
                // On attend l'ouverture animée avant de scroller
                setTimeout(() => {
                    const y = header.getBoundingClientRect().top + window.scrollY - 20;
                    window.scrollTo({ top: y, behavior: "smooth" });
                }, 350);
            }
        });
    });

    // Gestion du focus_id à l'affichage initial
    const focused = document.querySelector(".collapse.show");
    if (focused) {
        // On cherche le header qui l’a déclenché
        const header = focused.closest(".card").querySelector(".card-header");
        if (header) {
            const y = header.getBoundingClientRect().top + window.scrollY - 20;
            window.scrollTo({ top: y, behavior: "smooth" });
        }
    }
});