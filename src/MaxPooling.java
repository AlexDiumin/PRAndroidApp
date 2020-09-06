import java.math.BigDecimal;
import java.math.MathContext;

public class MaxPooling {

    private static final MathContext mathContext20 = new MathContext(20);

    private final TensorSize inputSize; // размер входного тензора
    private final TensorSize outputSize; // размер выходного тензора

    private final int scale; // во сколько раз уменьшаеться размерность

    private final Tensor mask; // бинарная маска для максимумов

    public MaxPooling(TensorSize inputSize, int scale) {
        this.inputSize = inputSize; // запоминаем входной размер

        // вычисляем выходной размер
        this.outputSize = new TensorSize(this.inputSize.depth,
                this.inputSize.height / scale,
                this.inputSize.width / scale);

        this.scale = scale; // запоминаем коэффициент уменьшения

        this.mask = new Tensor(this.inputSize); // создаем маску для максимумов
    }

    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(this.outputSize); // создаем выходной тензор

        BigDecimal max; // максимальный элемент подматрицы
        int hMax, wMax; // индексы максимального элемента подматрицы
        // проходимся по входному тензору
        for (int d = 0; d < input.getDepth(); d++) {
            for (int h = 0; h < input.getHeight(); h += this.scale) {
                for (int w = 0; w < input.getWidth(); w += this.scale) {
                    // начальное значение и индексы максимума (первый элемент подматрицы)
                    max = input.getByIndex(d, h, w);
                    hMax = h;
                    wMax = w;

                    // проходимся по подматрице и выбираем максимальный элемент
                    for (int i = h; i < h + this.scale; i++) {
                        for (int j = w; j < w + this.scale; j++) {
                            this.mask.setByIndex(d, i, j, BigDecimal.ZERO); // обнуляем маску

                            if (input.getByIndex(d, i, j).compareTo(max) > 0) {
                                max = input.getByIndex(d, i, j);
                                hMax = i;
                                wMax = j;
                            }
                        }
                    }

                    output.setByIndex(d, h / this.scale, w / this.scale, max); // записываем максимум в выходной тензор

                    this.mask.setByIndex(d, hMax, wMax, BigDecimal.ONE); // записываем 1 в маску на место максимального элемента
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    public Tensor backward(final Tensor deltasOut) {
        Tensor deltasIn = new Tensor(this.inputSize); // создаем тензор дельт для предыдущего слоя

        for (int d = 0; d < this.inputSize.depth; d++) {
            for (int h = 0; h < this.inputSize.height; h++) {
                for (int w = 0; w < this.inputSize.width; w++)
                    deltasIn.setByIndex(d, h, w, deltasOut.getByIndex(d, h / this.scale, w / this.scale)
                            .multiply(this.mask.getByIndex(d, h, w), mathContext20));
            }
        }

        return deltasIn; // возвращаем тензор дельт для предыдущего слоя
    }

    public final TensorSize getOutputSize() {
        return this.outputSize;
    }
}
