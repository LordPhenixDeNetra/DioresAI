import os
import ast
import pkg_resources

def extract_imports_from_file(file_path):
    """ Extrait les modules importés d'un fichier Python. """
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])  # Prendre seulement le module principal
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split('.')[0])

    return imports

def extract_imports_from_project(project_path):
    """ Parcours tous les fichiers Python du projet et extrait les modules importés. """
    all_imports = set()

    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                all_imports.update(extract_imports_from_file(file_path))

    print(f"Modules trouvés dans le projet : {all_imports}")  # Débogage
    return all_imports


def filter_installed_packages(imports):
    """ Vérifie si les modules sont installés via pip et récupère leur vrai nom. """
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    filtered_imports = {pkg for pkg in imports if pkg.lower() in installed_packages}
    return filtered_imports

if __name__ == "__main__":
    project_dir = "DioresAI/app"  # Dossier du projet (peut être modifié)
    imports = extract_imports_from_project(project_dir)
    filtered_imports = filter_installed_packages(imports)

    # Sauvegarde dans requirements.txt
    with open("requirements.txt", "w") as f:
        for package in sorted(filtered_imports):
            f.write(f"{package}\n")

    print("✅ Fichier requirements.txt généré avec succès !")
