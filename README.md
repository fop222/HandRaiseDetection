#HandRaiseDetection

CSE 310 – Final Project
Lehigh University
Forum Patel, Evan Mazor, Kevin Cawood, Lillian Mauger

HandRaiseDetection is a computer vision–based system designed to detect hand-raising gestures in classroom environments and discreetly alert instructors in real time. The project focuses on improving classroom accessibility and participation, particularly for instructors with visual impairments, by leveraging existing classroom camera infrastructure and real-time gesture recognition models. The system detects natural hand-raising motions without requiring students to press buttons, install software, or alter classroom behavior, maintaining a seamless teaching experience.

In large lecture halls, instructors—especially those with visual impairments—may unintentionally miss students raising their hands due to distance, occlusion, or limited peripheral vision. This project aims to:

*Improve inclusivity and equity in classroom participation
*Support instructors with visual impairments without requiring disclosure
*Preserve natural classroom interaction without added friction

The final system uses a hybrid detection pipeline:

*YOLOv8 for fast and accurate person detection
*MediaPipe Hands applied only within detected person bounding boxes
*Real-time visual and audio alerts for detected hand raises
