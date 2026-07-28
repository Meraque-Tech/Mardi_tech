import albumentations as A
import cv2
import os

class ImageAugmentor:
    def __init__(self, folder_path, num_augmentations):
        self.transform = A.Compose([
            A.OneOf([
                A.Sequential([
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-20, 20),
                        p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.35, 0.3),
                        contrast_limit=(-0.25, 0.2),
                        p=0.5
                    ),
                    A.Blur(blur_limit=(3, 3), p=0.2)
                ], p=0.8),
                A.Sequential([
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(-0.05, 0.05),
                        rotate=(-20, 20),
                        p=0.5
                    ),
                    A.OneOf([
                        A.RandomRain(
                            slant_range=(-10, 10),
                            drop_length=10,
                            drop_width=1,
                            drop_color=(180, 180, 180),
                            blur_value=3,
                            brightness_coefficient=0.75,
                            rain_type="drizzle",
                            p=1.0
                        ),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 0.5),
                            src_radius=70,
                            p=1.0
                        )
                    ], p=1.0),
                    A.Blur(blur_limit=(3, 3), p=0.2)
                ], p=0.2)
            ], p=1.0)
        ],
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.base_path = folder_path
        self.cnt = 0
        self.augmentation_src(self.base_path, num_augmentations)

    def find_image_path(self, label_path):
        label_dir = os.path.dirname(label_path)
        train_dir = os.path.dirname(label_dir)
        image_dir = os.path.join(train_dir, "images")
        base_name = os.path.splitext(os.path.basename(label_path))[0]

        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(image_dir, base_name + ext)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"No matching image file found for label: {label_path}")

    def augment_image_and_bboxes(self, label_path):
        image_path = self.find_image_path(label_path)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Failed to load image at: {image_path}")

        self.aug_label_path = os.path.splitext(label_path)[0] + f"_augmented_{self.cnt}.txt"
        self.aug_image_path = os.path.splitext(image_path)[0] + f"_augmented_{self.cnt}.jpg"

        with open(label_path, 'r') as f:
            lines = f.readlines()

        bboxes = []
        category_ids = []
        keypoints = []
        polygon_shapes = []

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            if len(parts) == 5:
                category_ids.append(class_id)
                x_center, y_center, width, height = map(float, parts[1:])
                bbox_album = self.yolo_to_xywh(image, x_center, y_center, width, height)
                bboxes.append(bbox_album)
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                coords = list(map(float, parts[1:]))
                num_points = len(coords) // 2
                polygon_shapes.append((class_id, num_points))
                for i in range(num_points):
                    x = coords[2*i] * image.shape[1]
                    y = coords[2*i+1] * image.shape[0]
                    keypoints.append((x, y))

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids, keypoints=keypoints)
        transformed['polygon_shapes'] = polygon_shapes
        return transformed

    def yolo_to_xywh(self, image, x_center, y_center, width, height):
        image_height, image_width = image.shape[:2]
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        bbox_width = width * image_width
        bbox_height = height * image_height
        return (x_min, y_min, bbox_width, bbox_height)

    def draw_yolo_box(self, frame, bounding_box, category_id):
        x, y, w, h = bounding_box
        center_x = x + w / 2
        center_y = y + h / 2

        yolo_box = [
            category_id,
            center_x / frame.shape[1],
            center_y / frame.shape[0],
            w / frame.shape[1],
            h / frame.shape[0]
        ]
        
        # Optionally draw the bbox
        x_draw = int(center_x - w / 2)
        y_draw = int(center_y - h / 2)
        w_draw = int(w)
        h_draw = int(h)
        cv2.rectangle(frame, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (0, 255, 0), 2)
        return yolo_box

    def augmet_data_dump(self, image, bboxes, category_ids, keypoints, polygon_shapes):
        cv2.imwrite(self.aug_image_path, image)
        img = image.copy()

        with open(self.aug_label_path, 'w') as txt_file:
            for bbox, category_id in zip(bboxes, category_ids):
                yolo_box = self.draw_yolo_box(img, bbox, category_id)
                txt_file.write(' '.join(map(str, yolo_box)) + '\n')
            
            pt_idx = 0
            image_h, image_w = image.shape[:2]
            for class_id, num_points in polygon_shapes:
                poly_data = [str(class_id)]
                for _ in range(num_points):
                    if pt_idx < len(keypoints):
                        kx, ky = keypoints[pt_idx]
                        kx_norm = min(max(kx / image_w, 0.0), 1.0)
                        ky_norm = min(max(ky / image_h, 0.0), 1.0)
                        poly_data.append(f"{kx_norm:.6f}")
                        poly_data.append(f"{ky_norm:.6f}")
                        pt_idx += 1
                txt_file.write(' '.join(poly_data) + '\n')

    def augmentation_src(self, folder_path, num_augmentations):
        self.cnt = 0
        train_folder = self.find_train_folder(folder_path)
        if train_folder is None:
            print(f"[INFO] No train/ folder found from: {folder_path}")
            print("✅ Batch augmentation done. Total augmented:", self.cnt)
            return

        label_dir = os.path.join(train_folder, "labels")
        image_dir = os.path.join(train_folder, "images")

        if not os.path.isdir(label_dir):
            print(f"[INFO] Missing labels folder: {label_dir}")
            print("✅ Batch augmentation done. Total augmented:", self.cnt)
            return

        if not os.path.isdir(image_dir):
            print(f"[INFO] Missing images folder: {image_dir}")
            print("✅ Batch augmentation done. Total augmented:", self.cnt)
            return

        label_files = [
            filename for filename in os.listdir(label_dir)
            if filename.endswith(".txt") and "_augmented_" not in filename
        ]
        total_expected = len(label_files) * num_augmentations

        print(f"[INFO] Train folder: {train_folder}")
        print(f"[INFO] Source labels: {len(label_files)}")
        print(f"[INFO] Augmentations per label: {num_augmentations}")
        print(f"[INFO] Expected augmented pairs: {total_expected}")

        if not label_files:
            print("✅ Batch augmentation done. Total augmented:", self.cnt)
            return

        for label_index, filename in enumerate(label_files, start=1):
            label_path = os.path.join(label_dir, filename)
            for augment_index in range(1, num_augmentations + 1):
                try:
                    transformed = self.augment_image_and_bboxes(label_path)
                    self.augmet_data_dump(
                        transformed['image'], 
                        transformed['bboxes'], 
                        transformed['category_ids'],
                        transformed.get('keypoints', []),
                        transformed.get('polygon_shapes', [])
                    )
                    self.cnt += 1

                    if self.cnt == total_expected or self.cnt % 50 == 0:
                        progress = (self.cnt / total_expected) * 100
                        print(f"[PROGRESS] {self.cnt}/{total_expected} ({progress:.1f}%)")
                except Exception as e:
                    print(f"[ERROR] Failed on {filename}: {e}")

            print(
                f"[PROGRESS] Finished label {label_index}/{len(label_files)}: "
                f"{filename}"
            )
        print("✅ Batch augmentation done. Total augmented:", self.cnt)

    def find_train_folder(self, folder_path):
        folder_path = os.path.abspath(folder_path)
        folder_name = os.path.basename(folder_path)

        if folder_name == "train":
            return folder_path

        if folder_name == "labels" and os.path.basename(os.path.dirname(folder_path)) == "train":
            return os.path.dirname(folder_path)

        train_folder = os.path.join(folder_path, "train")
        if os.path.isdir(train_folder):
            return train_folder

        return None


def main():
    src_path = "/home/aloy/palm_data_split"
    num_augmentations = 3
    augmentor = ImageAugmentor(src_path, num_augmentations)

if __name__ == "__main__":
    main()
