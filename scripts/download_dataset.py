#!/usr/bin/env python
"""
Download CBIS-DDSM dataset from Kaggle and prepare for Git LFS.
Run this script locally to download dataset and prepare for Git LFS tracking.
"""

import os
import sys
import kagglehub
from pathlib import Path
import shutil

def download_and_prepare():
    """Download CBIS-DDSM dataset and prepare for Git LFS."""
    
    print("=" * 60)
    print("DESCARGANDO DATASET CBIS-DDSM DESDE KAGGLE")
    print("=" * 60)
    
    # Directorios
    repo_root = Path(__file__).parent.parent
    raw_dir = repo_root / "breast-cancer-cnn-dashboard" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Directorio destino: {raw_dir}")
    
    # Descargar desde Kaggle
    print("\nDescargando dataset desde Kaggle...")
    print("Esto puede tomar varios minutos...")
    
    try:
        # Descargar dataset completo
        path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        print(f"Dataset descargado en cache de kagglehub: {path}")
        
        # Copiar archivos DICOM a data/raw/
        print("\nCopiando archivos DICOM a data/raw/...")
        copied = 0
        total_size = 0
        
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.dcm', '.dicom')):
                    src = Path(root) / f
                    dst = Path("breast-cancer-cnn-dashboard/data/raw") / f
                    shutil.copy2(src, dst)
                    size = src.stat().st_size
                    total_size += size
                    copied += 1
                    if copied % 100 == 0:
                        print(f"  Copiados {copied} archivos...")
        
        print(f"\n✅ Completado: {copied} archivos DICOM copiados")
        print(f"Tamaño total: {total_size / (1024**3):.2f} GB")
        
        # Verificar archivos
        raw_dir = Path("breast-cancer-cnn-dashboard/data/raw")
        files = list(raw_dir.rglob("*.dcm")) + list(raw_dir.rglob("*.DCM"))
        print(f"\nArchivos en data/raw/: {len(files)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def check_git_lfs():
    """Check if Git LFS is installed and configured."""
    import subprocess
    
    print("\n" + "=" * 60)
    print("VERIFICANDO GIT LFS")
    print("=" * 60)
    
    try:
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git LFS instalado: {result.stdout.strip()}")
            return True
        else:
            print("❌ Git LFS no instalado")
            return False
    except FileNotFoundError:
        print("❌ Git LFS no instalado")
        return False


def setup_git_lfs():
    """Configure Git LFS for DICOM files."""
    import subprocess
    
    print("\n" + "=" * 60)
    print("CONFIGURANDO GIT LFS PARA ARCHIVOS DICOM")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    # Inicializar Git LFS
    print("\nInicializando Git LFS...")
    subprocess.run(["git", "lfs", "install"], check=True)
    
    # Rastrear archivos .dcm
    print("Configurando tracking para .dcm y .DCM...")
    subprocess.run(["git", "lfs", "track", "*.dcm"], check=True)
    subprocess.run(["git", "lfs", "track", "*.DCM"], check=True)
    
    # Verificar .gitattributes
    gitattributes = Path(".gitattributes")
    if gitattributes.exists():
        content = gitattributes.read_text()
        print(f"\n✅ .gitattributes actualizado:")
        print(content)
    
    print("\n✅ Git LFS configurado correctamente")


def add_and_commit():
    """Add files to Git and commit."""
    import subprocess
    
    print("\n" + "=" * 60)
    print("AÑADIENDO ARCHIVOS A GIT")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    # Añadir .gitattributes primero
    print("\nAñadiendo .gitattributes...")
    subprocess.run(["git", "add", ".gitattributes"], check=True)
    
    # Añadir archivos DICOM
    print("Añadiendo archivos DICOM (esto puede tardar)...")
    result = subprocess.run(["git", "add", "breast-cancer-cnn-dashboard/data/raw/"], check=True)
    
    # Commit
    print("\nCreando commit...")
    subprocess.run([
        "git", "commit", "-m", 
        "Add CBIS-DDSM dataset DICOM files via Git LFS\n\n"
        "- Downloaded CBIS-DDSM dataset from Kaggle\n"
        "- Added DICOM files tracked with Git LFS\n"
        "- Dataset ready for Streamlit Cloud deployment"
    ], check=True)
    
    print("\n✅ Commit creado exitosamente")


def main():
    print("=" * 60)
    print("PREPARACIÓN DATASET CBIS-DDSM PARA STREAMLIT CLOUD")
    print("=" * 60)
    
    # 1. Verificar Git LFS
    if not check_git_lfs():
        print("\n❌ Git LFS no está instalado.")
        print("Instálalo con: git lfs install")
        print("En macOS: brew install git-lfs")
        print("En Ubuntu: sudo apt install git-lfs")
        print("En Windows: choco install git-lfs")
        return
    
    # 2. Configurar Git LFS
    setup_git_lfs()
    
    # 3. Descargar dataset
    if not download_and_prepare():
        print("\n❌ Error descargando dataset")
        return
    
    # 4. Añadir y commit
    try:
        add_and_commit()
    except Exception as e:
        print(f"\n❌ Error en commit: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ TODO LISTO PARA PUSH")
    print("=" * 60)
    print("\nEjecuta: git push origin main")
    print("Los archivos DICOM se subirán via Git LFS")
    print("Streamlit Cloud tendrá las imágenes disponibles directamente")


if __name__ == "__main__":
    main()