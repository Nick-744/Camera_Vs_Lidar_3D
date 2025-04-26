from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import os

class FillPaint:
    def __init__(self, root, masks_dir, rgb_dir):
        self.root = root
        self.masks_dir = masks_dir
        self.rgb_dir   = rgb_dir

        self.edited_masks_dir = os.path.join(masks_dir, '..', 'edited_masks')
        os.makedirs(self.edited_masks_dir, exist_ok = True)

        self.mask_files = sorted([
            f for f in os.listdir(masks_dir) if f.endswith('.png')
        ])
        self.index = 0
        self.scale_factor = 1.36 # Μεγέθυνση εικόνας

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.load_mask()
        self.canvas.bind("<Button-1>", self.handle_click)

        # btn_frame = tk.Frame(root)
        # btn_frame.pack()
        # tk.Button(btn_frame, text = "Save", command=self.save_mask).pack(side = tk.LEFT)
        # tk.Button(btn_frame, text = "Next", command=self.next_mask).pack(side = tk.LEFT)

        # Shortcut keys
        self.root.bind('<KeyPress-s>', lambda event: self.save_mask())
        self.root.bind('<KeyPress-S>', lambda event: self.save_mask())
        self.root.bind('<KeyPress-n>', lambda event: self.next_mask())
        self.root.bind('<KeyPress-N>', lambda event: self.next_mask())

        return;

    def load_mask(self):
        mask_path = os.path.join(self.masks_dir, self.mask_files[self.index])
        rgb_path  = os.path.join(self.rgb_dir, self.mask_files[self.index])

        self.mask = Image.open(mask_path).convert('RGBA')
        self.mask_array = np.array(self.mask)

        self.rgb = Image.open(rgb_path).convert('RGBA')
        
        self.root.title(f"Επεξεργασία: {self.mask_files[self.index]}")
        self.update_canvas()

        return;

    def update_canvas(self):
        edited_mask = Image.fromarray(self.mask_array)

        alpha = 128
        edited_mask.putalpha(alpha)

        composed = Image.alpha_composite(self.rgb.copy(), edited_mask)
        
        # --- Μεγέθυνση εικόνας ---
        new_size = (int(composed.width * self.scale_factor), int(composed.height * self.scale_factor))
        composed = composed.resize(new_size, Image.NEAREST) # NEAREST -> sharp pixels

        self.tk_image = ImageTk.PhotoImage(composed)
        self.canvas.config(width = self.tk_image.width(), height = self.tk_image.height())
        self.canvas.create_image(0, 0, anchor = tk.NW, image = self.tk_image)

        return;

    def handle_click(self, event):
        (x, y) = (int(event.x / self.scale_factor), int(event.y / self.scale_factor))
        if (0 <= x < self.mask_array.shape[1]) and (0 <= y < self.mask_array.shape[0]):
            target_color = tuple(self.mask_array[y, x])
            replacement_color = (255, 0, 0, 255)
            if target_color != replacement_color:
                self.flood_fill(x, y, target_color, replacement_color)
                self.update_canvas()
        
        return;

    def flood_fill(self, x, y, target_color, replacement_color):
        if target_color == replacement_color:
            return;
    
        stack = [(x, y)]
        while stack:
            (cx, cy) = stack.pop()
            if (0 <= cx < self.mask_array.shape[1]) and (0 <= cy < self.mask_array.shape[0]):
                current_color = tuple(self.mask_array[cy, cx])
                if current_color == target_color:
                    self.mask_array[cy, cx] = replacement_color
                    stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        
        return;

    def save_mask(self):
        save_path = os.path.join(self.edited_masks_dir, f'edited_{self.mask_files[self.index]}')
        Image.fromarray(self.mask_array).save(save_path)
        print(f'Saved: {save_path}')

        return;

    def next_mask(self):
        self.index += 1
        if self.index >= len(self.mask_files):
            print('ΤΕΛΟΣ!')
            self.root.quit()
        else:
            self.load_mask()

        return;

def main():
    base_dir  = os.path.dirname(__file__)
    masks_dir = os.path.join(base_dir, 'dataset', 'masks')
    rgb_dir   = os.path.join(base_dir, 'dataset', 'rgb')

    root = tk.Tk()
    FillPaint(root, masks_dir, rgb_dir)
    root.mainloop()

    return;

if __name__ == '__main__':
    main()
