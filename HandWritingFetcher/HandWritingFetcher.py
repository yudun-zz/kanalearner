import tkinter as tk

from PIL import Image, ImageDraw

from util.util import HANDWRITING_HIRAGANA_LABEL_LIST, HANDWRITING_KATAKANA_LABEL_LIST


class HandWritingFetcher(object):
    WHITE = (255, 255, 255)

    def __init__(self, width, height, img_output_path, stroke, resize=None):
        self.width = width
        self.height = height
        self.output_path = img_output_path
        self.stroke = stroke
        self.resize = resize

        self.mouse_state = "up"

        self.canvas = None
        self.xold = None
        self.yold = None
        self.image = None
        self.draw = None
        self.root = None

    def _save_image(self):
        if self.resize is not None:
            self.image = self.image.resize(self.resize)
        self.image.save(self.output_path)

    def _reset_image_and_draw(self):
        self.image = Image.new("RGB", (self.width, self.height), self.WHITE)
        self.draw = ImageDraw.Draw(self.image)

    def _hit_return(self, e):
        self._save_image()
        self.root.destroy()

    def _hit_esc(self, e):
        self.canvas.delete("all")
        self._reset_image_and_draw()

    def _mouse_down(self, e):
        self.mouse_state = "down"

    def _mouse_up(self, e):
        self.mouse_state = "up"
        self.xold = None
        self.yold = None

    def _paint(self, e):
        if self.mouse_state == "down":
            if self.xold is not None and self.yold is not None:
                self.canvas.create_line(self.xold, self.yold, e.x, e.y, smooth=tk.TRUE)
                self.draw.line([self.xold, self.yold, e.x, e.y], fill="black", width=self.stroke)
            self.xold = e.x
            self.yold = e.y

    def fetch(self):
        self.root = tk.Tk()

        # Tkinter create a canvas to draw on
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self._reset_image_and_draw()

        self.root.bind("<Return>", self._hit_return)
        self.root.bind("<Escape>", self._hit_esc)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)

        self.root.mainloop()


if __name__ == '__main__':
    for hiragana in HANDWRITING_HIRAGANA_LABEL_LIST[5:]:
        handWritingFetcher = HandWritingFetcher(200, 200,
                                                'hand_writing_data/' + hiragana + '.png', stroke=6, resize=(64, 64))
        handWritingFetcher.fetch()

    for katakana in HANDWRITING_KATAKANA_LABEL_LIST[:5]:
        handWritingFetcher = HandWritingFetcher(200, 200,
                                                'hand_writing_data/' + katakana + '.png', stroke=10, resize=(64, 64))
        handWritingFetcher.fetch()
