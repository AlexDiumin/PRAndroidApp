import java.math.BigDecimal;

public class Tensor {

    private final BigDecimal[][][] values;
    private final TensorSize size;

    public Tensor(int depth, int height, int width) {
        this.size = new TensorSize(depth, height, width);

        // инициализируем массив значений нулями
        this.values = new BigDecimal[this.size.depth][][];
        for (int d = 0; d < this.size.depth; d++) {
            this.values[d] = new BigDecimal[this.size.height][];
            for (int h = 0; h < this.size.height; h++) {
                this.values[d][h] = new BigDecimal[this.size.width];
                for (int w = 0; w < this.size.width; w++)
                    this.values[d][h][w] = BigDecimal.ZERO;
            }
        }
    }

    public Tensor(TensorSize size) {
        this.size = size;

        // инициализируем массив значений нулями
        this.values = new BigDecimal[this.size.depth][][];
        for (int d = 0; d < this.size.depth; d++) {
            this.values[d] = new BigDecimal[this.size.height][];
            for (int h = 0; h < this.size.height; h++) {
                this.values[d][h] = new BigDecimal[this.size.width];
                for (int w = 0; w < this.size.width; w++)
                    this.values[d][h][w] = BigDecimal.ZERO;
            }
        }
    }

    public void setByIndex(int d, int h, int w, BigDecimal value) {
        this.values[d][h][w] = value;
    }

    public final BigDecimal getByIndex(int d, int h, int w) {
        return this.values[d][h][w];
    }

    public final TensorSize getSize() {
        return this.size;
    }

    public final int getDepth() {
        return this.size.depth;
    }

    public final int getHeight() {
        return this.size.height;
    }

    public final int getWidth() {
        return this.size.width;
    }
}
