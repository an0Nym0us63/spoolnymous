module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/js/**/*.js",
    "./**/*.html" // pour attraper tout
  ],
  safelist: [
    // Bootstrap layout & grid
    'row', 'container', 'col', 'col-12', 'col-sm-6', 'col-lg-6', 'col-lg-3',
    // Typo
    'text-white', 'text-light', 'text-xs', 'text-sm', 'text-center', 'text-right', 'fw-bold',
    // Backgrounds
    'bg-dark', 'bg-gray-900', 'bg-gray-800', 'bg-primary', 'bg-warning',
    // Borders et ombres
    'border', 'border-t-0', 'border-warning', 'border-gray-700',
    'rounded', 'rounded-xl', 'rounded-t-xl', 'rounded-b-xl', 'shadow', 'shadow-md', 'shadow-lg',
    // Marges/paddings
    'mb-3', 'mb-4', 'mt-3', 'mt-4', 'p-3', 'p-4', 'px-4', 'py-2',
    // Cards
    'card', 'card-body', 'card-title', 'card-header', 'alert', 'alert-success', 'alert-dismissible', 'btn-close'
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}