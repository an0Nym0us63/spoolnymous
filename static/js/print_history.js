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

        $select.select2({
            width: '100%',
            tags: true,
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
                    return {
                        results: (data.results || []).map(g => ({ id: g.id, text: g.name }))
                    };
                },
                cache: true
            }
        }).on('select2:open', applyThemeToDropdown);

        setTimeout(() => $select.focus(), 100);
    }

    initSelect2();

    $('#filtersCollapse').on('shown.bs.collapse', function () {
        initSelect2();
    });

    $(document).on('shown.bs.modal', '.modal', function () {
        const $modal = $(this);
        const $select = $modal.find('.select2-ajax');
        if ($select.length) {
            initAjaxSelect2($select);
        }
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
