@echo off
REM Membuat struktur folder
mkdir recommender-system
cd recommender-system

mkdir 1-training
mkdir 2-deployment
mkdir 3-testing

REM Membuat file kosong di 1-training
echo.>1-training\config.py
echo.>1-training\train_model.py
echo.>1-training\requirements-training.txt

REM Membuat file kosong di 2-deployment
echo.>2-deployment\inference.py
echo.>2-deployment\requirements.txt
echo.>2-deployment\package_model.py

REM Membuat file kosong di 3-testing
echo.>3-testing\test_endpoint.py

echo Struktur folder dan file telah berhasil dibuat!
pause