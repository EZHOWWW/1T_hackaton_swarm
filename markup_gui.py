import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import io
import pyperclip  # For cross-platform clipboard


def estimate_reverse_affine_transform(world_coords, pixel_coords_square):
    A = []
    b = []
    for s, d in zip(pixel_coords_square, world_coords):
        u, v = s
        x, z = d
        A.append([u, v, 1, 0, 0, 0])
        A.append([0, 0, 0, u, v, 1])
        b.append(x)
        b.append(z)
    A = np.array(A)
    b = np.array(b)
    x_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x_sol.reshape((2, 3))


def apply_reverse_transform(pixel_coord, reverse_transform):
    u, v = pixel_coord
    transformed = np.dot(reverse_transform, np.array([u, v, 1]))
    return transformed


class ImageAnnotator:
    def __init__(self, master):
        self.master = master
        master.title("Coordinate Annotation")

        self.image_data = None
        self.image_tk = None
        self.canvas = tk.Canvas(master, width=600, height=400)
        self.canvas.pack(pady=5)
        self.canvas.bind("<Button-1>", self.add_point)

        self.points_square = []
        self.points_fires = []
        self.world_coords_square = [
            (-74.0, 78.0),
            (-74.0, 72.0),
            (-80.0, 72.0),
            (-80.0, 78.0),
        ]
        self.current_point_type = "square"
        self.user_height = tk.DoubleVar(master, value=0.0)

        tk.Label(master, text="Action:").pack()
        self.action_var = tk.StringVar(master)
        self.action_var.set("Mark Square (4 points)")
        self.action_var.trace_add("write", self.update_action)
        action_choices = ["Mark Square (4 points)", "Mark Fires"]
        action_menu = tk.OptionMenu(
            master, self.action_var, self.action_var.get(), *action_choices
        )
        action_menu.pack()

        tk.Label(master, text="Height (Y):").pack()
        height_entry = tk.Entry(master, textvariable=self.user_height)
        height_entry.pack()

        load_clipboard_button = tk.Button(
            master, text="Load from Clipboard", command=self.load_from_clipboard
        )
        load_clipboard_button.pack(pady=2)

        load_file_button = tk.Button(
            master, text="Load from File", command=self.load_from_file
        )
        load_file_button.pack(pady=2)

        clear_fires_button = tk.Button(
            master, text="Clear Fire Points", command=self.clear_fire_points
        )
        clear_fires_button.pack(pady=2)

        clear_all_button = tk.Button(
            master, text="Clear All Points", command=self.clear_all_points
        )
        clear_all_button.pack(pady=2)

        self.result_text = tk.Text(master, height=5, width=60)
        self.result_text.pack(pady=5)

        process_button = tk.Button(
            master, text="Calculate Coordinates", command=self.calculate_coordinates
        )
        process_button.pack(pady=5)

        copy_button = tk.Button(
            master, text="Copy to Clipboard", command=self.copy_to_clipboard
        )
        copy_button.pack(pady=5)

        self.calculated_real_coords = []  # Store the calculated coordinates

    def load_from_clipboard(self):
        try:
            clipboard_data = pyperclip.paste()
            if clipboard_data.startswith(
                b"\x89PNG\r\n\x1a\n"
            ) or clipboard_data.startswith(b"\xff\xd8\xff"):
                self.load_image_data(io.BytesIO(clipboard_data))
            else:
                messagebox.showerror(
                    "Error", "Clipboard does not contain a PNG or JPEG image."
                )
        except pyperclip.PyperclipException:
            messagebox.showerror(
                "Error",
                "Clipboard access failed. Ensure you have 'xclip' (Linux) or 'pyperclip' installed correctly.",
            )
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
        )
        if file_path:
            try:
                with open(file_path, "rb") as f:
                    self.load_image_data(io.BytesIO(f.read()))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image file: {e}")

    def load_image_data(self, image_stream):
        try:
            img = Image.open(image_stream)
            self.image_tk = ImageTk.PhotoImage(img)
            self.canvas.config(
                width=self.image_tk.width(), height=self.image_tk.height()
            )
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.points_square = []
            self.points_fires = []
            self.canvas.delete("point")
            self.result_text.delete(1.0, tk.END)
            self.calculated_real_coords = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image data: {e}")

    def update_action(self, *args):
        if self.action_var.get() == "Mark Square (4 points)":
            self.current_point_type = "square"
        else:
            self.current_point_type = "fire"

    def add_point(self, event):
        x, y = event.x, event.y
        if self.image_tk:
            if self.current_point_type == "square" and len(self.points_square) < 4:
                self.points_square.append((x, y))
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3, fill="red", tags="point"
                )
                if len(self.points_square) == 4:
                    messagebox.showinfo(
                        "Information", "All square points marked. Select 'Mark Fires'."
                    )
                    self.action_var.set("Mark Fires")
                    self.current_point_type = "fire"
            elif self.current_point_type == "fire":
                self.points_fires.append((x, y))
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3, fill="yellow", tags="point"
                )

    def clear_fire_points(self):
        self.points_fires = []
        self.canvas.delete("point")
        # Redraw the square points if they exist
        for x, y in self.points_square:
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="red", tags="point"
            )

    def clear_all_points(self):
        self.points_square = []
        self.points_fires = []
        self.canvas.delete("point")

    def calculate_coordinates(self):
        if len(self.points_square) != 4:
            messagebox.showerror("Error", "Please mark 4 points for the square.")
            return
        if not self.points_fires:
            messagebox.showinfo("Information", "Please mark at least one fire point.")
            return

        try:
            reverse_transform_matrix = estimate_reverse_affine_transform(
                self.world_coords_square, self.points_square
            )
            user_y = self.user_height.get()
            self.calculated_real_coords = []
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(
                tk.END, f"Real coordinates of fires (x, {user_y:.1f}, z):\n"
            )
            for i, pixel_coord in enumerate(self.points_fires):
                real_xz = apply_reverse_transform(pixel_coord, reverse_transform_matrix)
                real_coord = (real_xz[0], user_y, real_xz[1])
                self.calculated_real_coords.append(real_coord)
                self.result_text.insert(
                    tk.END,
                    f"Fire {i + 1}: x={real_coord[0]:.2f}, y={real_coord[1]:.1f}, z={real_coord[2]:.2f}\n",
                )

        except np.linalg.LinAlgError:
            messagebox.showerror(
                "Error",
                "Unable to calculate transformation. Ensure the square points are marked correctly.",
            )
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during calculation: {e}")

    def copy_to_clipboard(self):
        if not self.calculated_real_coords:
            messagebox.showinfo("Information", "Please calculate coordinates first.")
            return

        clipboard_string = "["
        for i, coord in enumerate(self.calculated_real_coords):
            clipboard_string += (
                f"Vector({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})"
            )
            if i < len(self.calculated_real_coords) - 1:
                clipboard_string += ", "
        clipboard_string += "]"

        try:
            pyperclip.copy(clipboard_string)
            messagebox.showinfo("Information", "Coordinates copied to clipboard!")
        except pyperclip.PyperclipException:
            messagebox.showerror(
                "Error",
                "Failed to copy to clipboard. Ensure 'pyperclip' is installed and working correctly.",
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
