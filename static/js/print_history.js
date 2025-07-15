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
            if (!$(this).hasClass('select2-hidden-accessible')) {
                $(this).select2({ width: '100%' }).on('select2:open', applyThemeToDropdown);
            }
        });
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