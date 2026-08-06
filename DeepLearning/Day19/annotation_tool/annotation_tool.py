import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class AnnotationTool:
    """A simple annotation tool for drawing bounding boxes on images."""

    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Tool")

        self.image_folder = ""
        self.image_files = []
        self.current_index = 0
        self.image_photo = None
        self.image_scale = 1.0
        self.image_display_width = 0
        self.image_display_height = 0
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.image_orig_width = 0
        self.image_orig_height = 0

        self.start_x = None
        self.start_y = None
        self.current_rect_id = None
        self.boxes_by_image = {}
        self.current_mode = "draw"
        self.selected_box_index = None
        self.current_action = None
        self.resize_handle = None
        self.last_drag_x = None
        self.last_drag_y = None

        self.class_options = ["Car", "Person", "Bike", "Truck", "Bus"]
        self.class_colors = {
            "Car": "#1f77b4",
            "Person": "#ff7f0e",
            "Bike": "#2ca02c",
            "Truck": "#d62728",
            "Bus": "#9467bd",
        }
        self.current_class = tk.StringVar(value=self.class_options[0])

        self.canvas_width = 1000
        self.canvas_height = 700

        self._build_ui()

    def _build_ui(self):
        """Build the user interface widgets."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        class_label = tk.Label(left_panel, text="Select Class", font=("Arial", 12, "bold"))
        class_label.pack(anchor=tk.W, pady=(0, 8))

        for class_name in self.class_options:
            radio_button = tk.Radiobutton(
                left_panel,
                text=class_name,
                value=class_name,
                variable=self.current_class,
                indicatoron=0,
                width=12,
                pady=5,
                bg=self.class_colors[class_name],
                fg="white",
                selectcolor=self.class_colors[class_name],
                activebackground=self.class_colors[class_name],
            )
            radio_button.pack(anchor=tk.W, pady=2)

        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(right_panel)
        control_frame.pack(fill=tk.X)

        tk.Button(control_frame, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Copy Previous Annotations", command=self.copy_previous_annotations).pack(side=tk.LEFT, padx=5)
        self.mode_button = tk.Button(control_frame, text="Mode: Draw", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(right_panel, text="No folder selected", font=("Arial", 11))
        self.status_label.pack(padx=10, pady=(10, 0), anchor=tk.W)

        self.canvas = tk.Canvas(right_panel, width=self.canvas_width, height=self.canvas_height, bg="lightgray")
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_draw)

        self.root.bind("<Key>", self._handle_keypress)
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<Left>", lambda event: self.previous_image())

    def open_folder(self):
        """Open a folder and load image files."""
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.image_folder = folder
        self.image_files = self._find_image_files(folder)
        self.current_index = 0
        self.boxes_by_image = {}
        self._ensure_labels_folder()

        if not self.image_files:
            self.status_label.config(text="No images found in selected folder.")
            self.canvas.delete("all")
            return

        self.show_image()

    def _find_image_files(self, folder):
        """Return sorted supported image file names in the folder."""
        supported_extensions = (".jpg", ".jpeg", ".png")
        return sorted(
            [filename for filename in os.listdir(folder)
             if filename.lower().endswith(supported_extensions)]
        )

    def _ensure_labels_folder(self):
        """Create a labels folder inside the image folder."""
        label_folder = os.path.join(self.image_folder, "labels")
        os.makedirs(label_folder, exist_ok=True)

    def _get_label_path(self, filename):
        """Return the label file path for a given image filename."""
        label_folder = os.path.join(self.image_folder, "labels")
        base_name, _ = os.path.splitext(filename)
        return os.path.join(label_folder, f"{base_name}.txt")

    def _save_annotations(self):
        """Save current image annotations in YOLO format."""
        if not self.image_files:
            return

        filename = self.image_files[self.current_index]
        label_path = self._get_label_path(filename)
        boxes = self.boxes_by_image.get(filename, [])

        with open(label_path, "w", encoding="utf-8") as label_file:
            for entry in boxes:
                class_name = entry["class"]
                coords = entry["coords"]
                class_id = self.class_options.index(class_name)
                yolo_box = self._convert_box_to_yolo(coords)
                label_file.write(f"{class_id} {yolo_box}\n")

    def _convert_box_to_yolo(self, coords):
        """Convert image-space coordinates to normalized YOLO values."""
        left, top, right, bottom = coords
        x_center = (left + right) / 2.0
        y_center = (top + bottom) / 2.0
        width = right - left
        height = bottom - top

        x_center /= self.image_orig_width
        y_center /= self.image_orig_height
        width /= self.image_orig_width
        height /= self.image_orig_height

        return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _load_annotations(self, filename):
        """Load existing YOLO annotations for the current image."""
        label_path = self._get_label_path(filename)
        boxes = []

        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        continue

                    if class_id < 0 or class_id >= len(self.class_options):
                        continue

                    class_name = self.class_options[class_id]
                    coords = self._convert_yolo_to_box((x_center, y_center, width, height))
                    boxes.append({"coords": coords, "class": class_name})

        self.boxes_by_image[filename] = boxes

    def _convert_yolo_to_box(self, yolo_values):
        """Convert normalized YOLO values back to image-space coordinates."""
        x_center, y_center, width, height = yolo_values
        left = (x_center - width / 2.0) * self.image_orig_width
        top = (y_center - height / 2.0) * self.image_orig_height
        right = (x_center + width / 2.0) * self.image_orig_width
        bottom = (y_center + height / 2.0) * self.image_orig_height
        return self._normalize_box(left, top, right, bottom)

    def show_image(self):
        """Display the current image and its bounding boxes."""
        if not self.image_files:
            return

        filename = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, filename)

        self._load_image(image_path)
        self._load_annotations(filename)
        self.selected_box_index = None
        self._render_image()
        self._render_boxes()
        self._update_status()

    def _load_image(self, image_path):
        """Load, resize, and store the current image."""
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        self.image_orig_width = orig_width
        self.image_orig_height = orig_height

        width_ratio = self.canvas_width / orig_width
        height_ratio = self.canvas_height / orig_height
        self.image_scale = min(width_ratio, height_ratio, 1.0)

        self.image_display_width = int(orig_width * self.image_scale)
        self.image_display_height = int(orig_height * self.image_scale)
        self.image_offset_x = (self.canvas_width - self.image_display_width) // 2
        self.image_offset_y = (self.canvas_height - self.image_display_height) // 2

        image = image.resize((self.image_display_width, self.image_display_height), Image.Resampling.LANCZOS)
        self.image_photo = ImageTk.PhotoImage(image)

    def _render_image(self):
        """Clear the canvas and draw the current image centered."""
        self.canvas.delete("all")
        self.canvas.create_image(self.image_offset_x, self.image_offset_y, image=self.image_photo, anchor=tk.NW)

    def _render_boxes(self):
        """Draw saved boxes for the current image."""
        filename = self.image_files[self.current_index]
        boxes = self.boxes_by_image.get(filename, [])
        for index, entry in enumerate(boxes):
            self._draw_box_entry(entry, index == self.selected_box_index)

    def _draw_box_entry(self, entry, selected=False):
        """Draw a single box entry on the canvas."""
        coords = entry["coords"]
        class_name = entry["class"]
        color = self.class_colors.get(class_name, "red")
        canvas_coords = self._image_to_canvas_coords(coords)
        outline_width = 3 if selected else 2
        outline_color = "yellow" if selected else color

        self.canvas.create_rectangle(*canvas_coords, outline=outline_color, width=outline_width)
        text_x = canvas_coords[0] + 4
        text_y = canvas_coords[1] - 16
        self.canvas.create_text(text_x, text_y, anchor=tk.NW, text=class_name, fill=color, font=("Arial", 10, "bold"))

        if selected:
            for handle_coords in self._get_corner_handles(coords):
                handle_canvas = self._image_to_canvas_coords(handle_coords)
                self.canvas.create_rectangle(*handle_canvas, fill="white", outline="black")

    def _update_status(self):
        """Update the filename, image count, annotation count, and mode."""
        filename = self.image_files[self.current_index] if self.image_files else "No folder selected"
        total = len(self.image_files)
        annotations = len(self.boxes_by_image.get(self.image_files[self.current_index], [])) if self.image_files else 0
        self.status_label.config(text=f"{filename}    ({self.current_index + 1}/{total})    Annotations: {annotations}    Mode: {self.current_mode.capitalize()}")

    def start_draw(self, event):
        """Handle a canvas click based on the current mode."""
        if not self.image_files:
            return

        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
        if self.current_mode == "select":
            self._prepare_select_action(image_x, image_y)
            return

        self.start_x = image_x
        self.start_y = image_y
        self.selected_box_index = None
        self.current_action = None
        color = self.class_colors.get(self.current_class.get(), "red")
        canvas_start = self._image_to_canvas_coords((self.start_x, self.start_y, self.start_x, self.start_y))
        self.current_rect_id = self.canvas.create_rectangle(
            *canvas_start,
            outline=color,
            width=2,
        )

    def draw_box(self, event):
        """Update the temporary bounding box while dragging."""
        if self.current_mode == "select" and self.current_action in {"move", "resize"}:
            image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
            self._update_selected_box(image_x, image_y)
            self._render_image()
            self._render_boxes()
            return

        if self.current_rect_id is None:
            return

        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
        canvas_coords = self._image_to_canvas_coords(self._normalize_box(self.start_x, self.start_y, image_x, image_y))
        self.canvas.coords(self.current_rect_id, *canvas_coords)

    def finish_draw(self, event):
        """Finalize the bounding box and store it."""
        if self.current_mode == "select" and self.current_action in {"move", "resize"}:
            self.current_action = None
            self.resize_handle = None
            self.last_drag_x = None
            self.last_drag_y = None
            self._update_status()
            return

        if self.current_rect_id is None:
            return

        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
        box = self._normalize_box(self.start_x, self.start_y, image_x, image_y)
        self._store_box(box)
        self.current_rect_id = None

    def _store_box(self, box):
        """Save the drawn box and class label for the current image."""
        filename = self.image_files[self.current_index]
        class_name = self.current_class.get()
        self.boxes_by_image.setdefault(filename, []).append({
            "coords": box,
            "class": class_name,
        })

    def select_box(self, event):
        """Select a box by clicking inside it."""
        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
        filename = self.image_files[self.current_index]
        boxes = self.boxes_by_image.get(filename, [])
        selected_index = self._find_box_at_point(boxes, image_x, image_y)
        self.selected_box_index = selected_index
        self.current_action = None
        self._render_image()
        self._render_boxes()
        self._update_status()

    def _prepare_select_action(self, image_x, image_y):
        """Determine whether the user wants to move or resize in select mode."""
        filename = self.image_files[self.current_index]
        boxes = self.boxes_by_image.get(filename, [])
        if self.selected_box_index is not None:
            selected_box = boxes[self.selected_box_index]
            handle = self._handle_at_point(selected_box["coords"], image_x, image_y)
            if handle:
                self.current_action = "resize"
                self.resize_handle = handle
                self.last_drag_x = image_x
                self.last_drag_y = image_y
                return

            if self._point_in_box(selected_box["coords"], image_x, image_y):
                self.current_action = "move"
                self.last_drag_x = image_x
                self.last_drag_y = image_y
                return

        selected_index = self._find_box_at_point(boxes, image_x, image_y)
        self.selected_box_index = selected_index
        if selected_index is not None:
            self.current_action = "move"
            self.last_drag_x = image_x
            self.last_drag_y = image_y
        else:
            self.current_action = None
            self.resize_handle = None

    def _find_box_at_point(self, boxes, x, y):
        """Return the index of the first box that contains the point."""
        for i, entry in enumerate(boxes):
            left, top, right, bottom = entry["coords"]
            if left <= x <= right and top <= y <= bottom:
                return i
        return None

    def _point_in_box(self, coords, x, y):
        """Return true if a point is inside the box."""
        left, top, right, bottom = coords
        return left <= x <= right and top <= y <= bottom

    def _handle_at_point(self, coords, x, y):
        """Return the resize handle name under the point, if any."""
        for direction, handle_coords in self._get_corner_handles(coords).items():
            left, top, right, bottom = handle_coords
            if left <= x <= right and top <= y <= bottom:
                return direction
        return None

    def _get_corner_handles(self, coords):
        """Return small corner handle boxes in image coordinates."""
        left, top, right, bottom = coords
        size = 8 / self.image_scale
        return {
            "nw": (left - size, top - size, left + size, top + size),
            "ne": (right - size, top - size, right + size, top + size),
            "se": (right - size, bottom - size, right + size, bottom + size),
            "sw": (left - size, bottom - size, left + size, bottom + size),
        }

    def delete_selected_box(self):
        """Delete the selected box from the current image."""
        filename = self.image_files[self.current_index] if self.image_files else None
        if filename is None or self.selected_box_index is None:
            return

        boxes = self.boxes_by_image.get(filename, [])
        if 0 <= self.selected_box_index < len(boxes):
            boxes.pop(self.selected_box_index)
            self.selected_box_index = None
            self._render_image()
            self._render_boxes()
            self._update_status()

    def set_mode(self, mode):
        """Set the interaction mode and update the UI."""
        if mode not in {"draw", "select"}:
            return

        self.current_mode = mode
        self.mode_button.config(text=f"Mode: {mode.capitalize()}")
        if mode == "draw":
            self.selected_box_index = None
        self.current_action = None
        self.resize_handle = None
        self.last_drag_x = None
        self.last_drag_y = None
        self._update_status()

    def toggle_mode(self):
        """Toggle between draw and select mode."""
        self.set_mode("select" if self.current_mode == "draw" else "draw")

    def _handle_keypress(self, event):
        """Respond to keyboard shortcuts."""
        key = event.keysym.lower()
        if key == "d":
            self.set_mode("draw")
        elif key == "s":
            self.set_mode("select")
        elif key == "delete":
            self.delete_selected_box()

    def _update_selected_box(self, image_x, image_y):
        """Move or resize the selected box while dragging."""
        if self.selected_box_index is None:
            return

        filename = self.image_files[self.current_index]
        boxes = self.boxes_by_image.get(filename, [])
        if not (0 <= self.selected_box_index < len(boxes)):
            return

        coords = boxes[self.selected_box_index]["coords"]
        if self.current_action == "move":
            dx = image_x - self.last_drag_x
            dy = image_y - self.last_drag_y
            boxes[self.selected_box_index]["coords"] = self._move_box(coords, dx, dy)
            self.last_drag_x = image_x
            self.last_drag_y = image_y
        elif self.current_action == "resize" and self.resize_handle is not None:
            boxes[self.selected_box_index]["coords"] = self._resize_box(coords, image_x, image_y, self.resize_handle)
            self.last_drag_x = image_x
            self.last_drag_y = image_y

    def _move_box(self, coords, dx, dy):
        """Move a box by dx/dy within image bounds."""
        left, top, right, bottom = coords
        left += dx
        right += dx
        top += dy
        bottom += dy

        width = right - left
        height = bottom - top

        left = max(0, min(left, self.image_orig_width - width))
        top = max(0, min(top, self.image_orig_height - height))
        right = left + width
        bottom = top + height

        return left, top, right, bottom

    def _resize_box(self, coords, x, y, handle):
        """Resize a box using the selected handle."""
        left, top, right, bottom = coords
        if handle == "nw":
            left = min(max(0, x), right - 1)
            top = min(max(0, y), bottom - 1)
        elif handle == "ne":
            right = max(min(self.image_orig_width, x), left + 1)
            top = min(max(0, y), bottom - 1)
        elif handle == "se":
            right = max(min(self.image_orig_width, x), left + 1)
            bottom = max(min(self.image_orig_height, y), top + 1)
        elif handle == "sw":
            left = min(max(0, x), right - 1)
            bottom = max(min(self.image_orig_height, y), top + 1)

        return self._normalize_box(left, top, right, bottom)

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image space and clamp inside the image."""
        image_x = canvas_x - self.image_offset_x
        image_y = canvas_y - self.image_offset_y
        image_x = max(0, min(image_x, self.image_display_width))
        image_y = max(0, min(image_y, self.image_display_height))
        return image_x, image_y

    def _image_to_canvas_coords(self, coords):
        """Convert image-space coordinates to canvas coordinates."""
        left, top, right, bottom = coords
        return (
            left + self.image_offset_x,
            top + self.image_offset_y,
            right + self.image_offset_x,
            bottom + self.image_offset_y,
        )

    def _normalize_box(self, x1, y1, x2, y2):
        """Normalize coordinates so the box is top-left to bottom-right."""
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        return left, top, right, bottom

    def copy_previous_annotations(self):
        """Copy annotations from the previous image if the current image is empty."""
        if not self.image_files or self.current_index == 0:
            return

        current_filename = self.image_files[self.current_index]
        previous_filename = self.image_files[self.current_index - 1]
        current_boxes = self.boxes_by_image.get(current_filename, [])
        previous_boxes = self.boxes_by_image.get(previous_filename, [])

        if current_boxes or not previous_boxes:
            return

        copied = [
            {"coords": tuple(box["coords"]), "class": box["class"]}
            for box in previous_boxes
        ]
        self.boxes_by_image[current_filename] = copied
        self.selected_box_index = None
        self._render_image()
        self._render_boxes()
        self._update_status()

    def next_image(self):
        """Move to the next image in the folder."""
        if self.current_index + 1 < len(self.image_files):
            self._save_annotations()
            self.current_index += 1
            self.show_image()

    def previous_image(self):
        """Move to the previous image in the folder."""
        if self.current_index > 0:
            self._save_annotations()
            self.current_index -= 1
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    AnnotationTool(root)
    root.mainloop()
