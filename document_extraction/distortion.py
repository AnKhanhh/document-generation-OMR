import cv2
import numpy as np


class DocumentDistorter:
    """
    Apply distortions to images, simulating camera distortion
    """

    @staticmethod
    def apply_perspective_distortion(image: np.ndarray, severity: float = 0.2) -> np.ndarray:
        """
        Applies perspective distortion
        Args:
            image: Input BGR image
            severity: (0.0 - 1.0). Higher value means sharper angle
        Returns:
            distorted image
        """
        h, w = image.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        # Calculate safe displacement limits
        output_w, output_h = int(w * 1.2), int(h * 1.2)
        margin_x = (output_w - w) // 2
        margin_y = (output_h - h) // 2

        # Center the image in the output frame
        centered_dst = np.float32([
            [margin_x, margin_y],
            [margin_x + w, margin_y],
            [margin_x, margin_y + h],
            [margin_x + w, margin_y + h]
        ])

        # Calculate max safe displacement that won't push corners outside the output frame
        max_x_disp = min(margin_x, int(w * severity * 0.5))
        max_y_disp = min(margin_y, int(h * severity * 0.5))

        # Generate random displacements within safe limits
        rng = np.random.RandomState(42)
        corner_displacements = [
            [rng.randint(-max_x_disp, max_x_disp), rng.randint(-max_y_disp, max_y_disp)],
            [rng.randint(-max_x_disp, max_x_disp), rng.randint(-max_y_disp, max_y_disp)],
            [rng.randint(-max_x_disp, max_x_disp), rng.randint(-max_y_disp, max_y_disp)],
            [rng.randint(-max_x_disp, max_x_disp), rng.randint(-max_y_disp, max_y_disp)]
        ]

        # Apply displacements to the centered destination points
        dst_points = np.float32([
            [centered_dst[0][0] + corner_displacements[0][0], centered_dst[0][1] + corner_displacements[0][1]],
            [centered_dst[1][0] + corner_displacements[1][0], centered_dst[1][1] + corner_displacements[1][1]],
            [centered_dst[2][0] + corner_displacements[2][0], centered_dst[2][1] + corner_displacements[2][1]],
            [centered_dst[3][0] + corner_displacements[3][0], centered_dst[3][1] + corner_displacements[3][1]]
        ])

        # Validate that all points are within output bounds (with a small safety margin)
        safety_margin = 2  # pixels
        for point in dst_points:
            x, y = point
            if x < safety_margin or x >= output_w - safety_margin or y < safety_margin or y >= output_h - safety_margin:
                # If any point is too close to the edge, reduce displacements and recalculate
                for i in range(4):
                    corner_displacements[i][0] //= 2
                    corner_displacements[i][1] //= 2

                dst_points = np.float32([
                    [centered_dst[0][0] + corner_displacements[0][0], centered_dst[0][1] + corner_displacements[0][1]],
                    [centered_dst[1][0] + corner_displacements[1][0], centered_dst[1][1] + corner_displacements[1][1]],
                    [centered_dst[2][0] + corner_displacements[2][0], centered_dst[2][1] + corner_displacements[2][1]],
                    [centered_dst[3][0] + corner_displacements[3][0], centered_dst[3][1] + corner_displacements[3][1]]
                ])
                break

        # Apply perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        distorted = cv2.warpPerspective(image, perspective_matrix, (output_w, output_h))

        return distorted

    @staticmethod
    def apply_lighting_variation(image: np.ndarray, brightness_factor: float = 1.0,
                                 contrast_factor: float = 1.0,
                                 light_direction: str | None = 'random',
                                 max_shadow: float = 0.3) -> np.ndarray:
        """
        Apply distortions for brightness, contrast, and directional lighting
        Args:
            image: BGR image
            brightness_factor: Multiplied with pixel values
            contrast_factor: Multiplied with pixel difference from mean
            light_direction: source of light direction
            max_shadow: higher value mean heavier shadow  (0.0 - 1.0)

        Returns:
            Image with modified lighting
        """
        # Convert to float for calculations
        result = image.astype(np.float32)

        # Apply brightness and contrast
        if brightness_factor != 1.0:
            result = result * brightness_factor
        if contrast_factor != 1.0:
            mean = np.mean(result, axis=(0, 1), keepdims=True)
            result = mean + (result - mean) * contrast_factor

        # Apply lighting
        if light_direction is not None and max_shadow != 0:
            h, w = image.shape[:2]
            if light_direction == 'random':
                light_direction = np.random.choice(['top', 'bottom', 'left', 'right'])
            # Create lighting gradient vector based on direction
            if light_direction == 'top':
                gradient = np.linspace(1.0, 1.0 - max_shadow, h)[:, np.newaxis]
            elif light_direction == 'bottom':
                gradient = np.linspace(1.0 - max_shadow, 1.0, h)[:, np.newaxis]
            elif light_direction == 'left':
                gradient = np.linspace(1.0, 1.0 - max_shadow, w)[np.newaxis, :]
            elif light_direction == 'right':
                gradient = np.linspace(1.0 - max_shadow, 1.0, w)[np.newaxis, :]

            # Apply gradient
            gradient = np.expand_dims(gradient, axis=2)
            result = result * gradient

        # Clip values, convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applies Gaussian blur to simulate camera focus.
        Args:
            image: BGR image
            kernel_size: Kernel size. Larger kernel for stronger blur

        Returns:
            Blurred image
        """
        # Ensure kernel size is odd
        kernel_size = kernel_size | 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)  # auto sigma

    @staticmethod
    def apply_noise(image: np.ndarray, amount: float = 0.05) -> np.ndarray:
        """
        Applies Gaussian noise
        Args:
            image: BGR image
            amount: (0 - 1). Noise strength

        Returns:
            Noisy image
        """
        noise = np.zeros_like(image, dtype=np.float32)
        std = amount * 255
        cv2.randn(noise, 0, std)

        # Add noise to image with automatic clipping
        result = cv2.add(image, noise, dtype=cv2.CV_8U)
        return result

    @staticmethod
    def apply_rotation(image: np.ndarray, angle: float = 0.0) -> np.ndarray:
        """
        Rotates the image by the specified angle.
        Args:
            image: Input BGR image
            angle: Rotation angle in degrees
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image dimensions to ensure we don't crop the image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
        return rotated
