import argostranslate.package
import argostranslate.translate

_initialized = False

def initialize():
    """
    Installe et configure le modèle de traduction en<->fr s'il n'est pas déjà installé.
    """
    global _initialized
    if _initialized:
        return

    # Met à jour l'index des modèles disponibles
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    # Vérifie si le modèle est déjà installé
    installed_languages = argostranslate.translate.get_installed_languages()
    en_installed = any(lang.code == 'en' for lang in installed_languages)
    fr_installed = any(lang.code == 'fr' for lang in installed_languages)

    if en_installed and fr_installed:
        _initialized = True
        return

    # Sinon, télécharge et installe le modèle en→fr
    for package in available_packages:
        if package.from_code == 'en' and package.to_code == 'fr':
            path = package.download()
            argostranslate.package.install_from_path(path)
            break

    _initialized = True


def translate_title(title_en: str) -> str:
    """
    Traduit une chaîne anglaise en français.

    Args:
        title_en (str): texte en anglais.

    Returns:
        str: texte traduit en français.
    """
    initialize()
    return argostranslate.translate.translate(title_en, 'en', 'fr')