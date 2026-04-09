@echo off
title Advanced Skin Disease Detection AI
echo ========================================
echo 🩺 Installing Clinical AI System...
echo ========================================
pip install -r requirements.txt

echo ========================================
echo 📊 Creating Test Dataset (400 images)...
echo ========================================
python quick_test_data.py

echo ========================================
echo 🎯 Training Model (2 minutes)...
echo ========================================
python train.py

echo ========================================
echo 🌐 Launching Advanced Web App...
echo ========================================
python app.py

echo.
echo 🎉 SUCCESS! Open browser to the Gradio link!
pause
