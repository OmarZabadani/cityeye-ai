"""This module serves as the main entry point for the vehicle counting application. 
It orchestrates the entire pipeline, from loading images to processing them and counting vehicles."""

import os
import cv2

from pipeline.process_image import process_number_of_cars_and_types


def save_annotated_image(image_path, output_path, vehicle_count, vehicle_type_counts):
    """
    Save an annotated image with vehicle count and types.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    # Draw total vehicle count
    cv2.putText(
        img,
        f"Total Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Draw vehicle type counts
    y_offset = 80
    for v_type, count in vehicle_type_counts.items():
        cv2.putText(
            img,
            f"{v_type}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        y_offset += 30

    # Save output
    cv2.imwrite(output_path, img)
    print(f"[SAVED] {output_path}")


def process_images(image_paths, output_dir):
    """
    Process multiple images and save annotated outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        print(f"\n[PROCESSING] {image_path}")

        result = process_number_of_cars_and_types(image_path)

        if result is None:
            print("[SKIPPED] Invalid image")
            continue

        vehicle_count, vehicle_types, vehicle_type_counts = result

        print(f"[RESULT] Total Vehicles: {vehicle_count}")
        print(f"[DETAILS] {vehicle_type_counts}")

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"output_{filename}")

        save_annotated_image(
            image_path,
            output_path,
            vehicle_count,
            vehicle_type_counts
        )


def main():
    """
    Entry point of the traffic analysis system.
    """

    image_paths = [
        r"D:\download\uni\year4\term2\GP2\Data\images\bbb.jpg",
        r"D:\download\uni\year4\term2\GP2\Data\images\Screenshot 2026-04-13 212234.png",
        r"D:\download\uni\year4\term2\GP2\Data\images\Screenshot 2026-04-15 185210.png"
    ]

    output_dir = r"D:\download\uni\year4\term2\GP2\Data\output"

    process_images(image_paths, output_dir)


if __name__ == "__main__":
    main()