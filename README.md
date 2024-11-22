
# BuildSafe-AI

## Overview

SafeSiteAI is a Python-based application designed to enhance construction site safety by detecting whether workers are wearing hard hats in real-time. The system uses a YOLOv8 model fine-tuned with site-specific data and processes live video feeds to identify safety violations.

## Features
Real-Time Detection: Identifies workers wearing or not wearing hard hats.
Custom Dataset: Fine-tuned with data collected from construction sites to improve accuracy in real-world conditions.
Flexible Deployment: Compatible with IP cameras and supports various resolutions.
Notification System: Alerts users when a hard hat violation is detected (future enhancement).

## Project Highlights

Collected over 1000 site-specific frames and augmented 200+ images for robust training.
Fine-tuned a YOLOv8 model pretrained on a dataset of 5,000 images.
Implemented a Python-based inference pipeline with OpenCV for live video stream analysis.
Deployed a scalable application leveraging real-time object detection.
## Authors

- [@Abdullah Hamid](https://www.github.com/abdullah-hamid)

