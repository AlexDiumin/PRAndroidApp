public class TensorSize {

    public int depth; // глубина
    public int height; // высота
    public int width; // ширина

    public TensorSize() {
        this.depth = 0;
        this.height = 0;
        this.width = 0;
    }

    public TensorSize(int depth, int height, int width) {
        this.depth = depth;
        this.height = height;
        this.width = width;
    }
}
