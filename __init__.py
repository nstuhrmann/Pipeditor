"""
Beim Import dieses Pakets werden automatisch alle .py-Dateien in diesem
Ordner importiert (außer __init__.py selbst), damit deren @register_step-
Decorators laufen und die Steps in STEP_REGISTRY landen.

Neue Algorithmen-Datei einfach hier in den Ordner legen - sie wird
automatisch gefunden, ohne dass man irgendwo manuell einen Import
hinzufügen muss.
"""
import pkgutil
import importlib

_package_dir = __path__
_package_name = __name__

for _, module_name, _ in pkgutil.iter_modules(_package_dir):
    importlib.import_module(f"{_package_name}.{module_name}")
