import java.math.BigDecimal;
import java.util.Vector;

public class Tensor {

    private TensorSize size; // размерность тензора
    private Vector<BigDecimal> values; // значения тензора

    private int dw; // произведения глубины на ширину для индексации

    // инициализация по размерам
    private void init(int width, int height, int depth) {
        size = new TensorSize(depth, height, width);

        dw = depth*width;

        values = new Vector<>(width*height*depth);
        for (int i = 0; i < values.capacity(); i++)
            values.add(BigDecimal.ZERO);
//            values.set(i, BigDecimal.ZERO);
    }

    public Tensor(int width, int height, int depth) {
        init(width, height, depth);
    }

    public Tensor(final TensorSize tensorSize) {
        init(tensorSize.width, tensorSize.height, tensorSize.depth);
    }

    public final BigDecimal getByIndex(int d, int h, int w) {
        return values.get(h*dw + w*size.depth + d);
    }

    public final BigDecimal getByIndex(int i) {
        return values.get(i);
    }

    public void setByIndex(int d, int h, int w, BigDecimal value) {
        values.set(h*dw + w*size.depth + d, value);
    }

    public void setByIndex(int i, BigDecimal value) {
        values.set(i, value);
    }

    public final TensorSize getSize() {
        return size;
    }

    public void print(final Tensor tensor) {
        for (int d = 0; d < tensor.size.depth; d++) {
            for (int h = 0; h < tensor.size.height; h++) {
                for (int w = 0; w < tensor.size.width; w++)
                    System.out.print(tensor.values.get(h*tensor.dw + w*tensor.size.depth + d) + " ");

                System.out.println();
            }
            System.out.println();
        }
    }

    public void print() {
        final Tensor tensor = this;
        for (int d = 0; d < tensor.size.depth; d++) {
            for (int h = 0; h < tensor.size.height; h++) {
                for (int w = 0; w < tensor.size.width; w++)
                    System.out.print(tensor.values.get(h* tensor.dw + w*tensor.size.depth + d) + " ");

                System.out.println();
            }
            System.out.println();
        }
    }

    public static Tensor arrayToTensor(Tensor[] tensorsArray) {
        Tensor result = new Tensor(tensorsArray.length, 1, 1);
        for (int i = 0; i < tensorsArray.length; i++)
            result.setByIndex(i, tensorsArray[i].getByIndex(0));
        return result;
    }
}
