## HandRaiseDetection

## CSE 310 – Final Project  
**Lehigh University**  
**Authors:** Forum Patel, Evan Mazor, Kevin Cawood, Lillian Mauger

---

## Overview
**HandRaiseDetection** is a computer vision–based system designed to detect hand-raising gestures in classroom environments and discreetly alert instructors in real time. The project focuses on improving classroom accessibility and participation—particularly for instructors with visual impairments—by leveraging existing classroom camera infrastructure and real-time gesture recognition models.

The system detects natural hand-raising motions without requiring students to press buttons, install software, or alter classroom behavior, preserving a seamless and inclusive teaching experience.

---

## Motivation
In large lecture halls, instructors—especially those with visual impairments—may unintentionally miss students raising their hands due to distance, occlusion, or limited peripheral vision. This project aims to:

- Improve inclusivity and equity in classroom participation  
- Support instructors with visual impairments without requiring disclosure  
- Preserve natural classroom interaction without added friction  

---

## System Design
The final system uses a **hybrid detection pipeline** that balances accuracy, speed, and scalability:

- **YOLOv8** for fast and accurate person detection  
- **MediaPipe Hands** applied only within detected person bounding boxes  
- **Real-time visual and audio alerts** to notify instructors of detected hand raises  

This approach reduces false positives, improves responsiveness, and remains practical for real-world classroom environments.

---

## Key Features
- Real-time hand-raise detection  
- No required student interaction or software installation  
- Accessible, high-contrast visual alerts  
- Optional audio cues to reduce visual scanning  

---

## Technologies Used
- Python  
- OpenCV  
- YOLOv8  
- MediaPipe Hands  

---

## Disclaimer
This project was developed as an academic prototype for CSE 310 and is intended for educational and experimental purposes.
