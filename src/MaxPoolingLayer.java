import java.math.BigDecimal;
import java.math.MathContext;

public class MaxPoolingLayer {

    private final MathContext mathContext20 = new MathContext(20);


    private TensorSize inputSize = new TensorSize(); // размер входа
    private TensorSize outputSize = new TensorSize(); // размер выхода

    private int scale; // во сколько раз уменьшаеться размерность

    private Tensor mask; // бинарная маска с положениями максимумов

    // создание слоя
    public MaxPoolingLayer(TensorSize tensorSize) {
        int scale = 2;

        // запоминаем входной размер
        inputSize.width = tensorSize.width;
        inputSize.height = tensorSize.height;
        inputSize.depth = tensorSize.depth;

        // вычисляем выходной размер
        outputSize.width = tensorSize.width / scale;
        outputSize.height = tensorSize.height / scale;
        outputSize.depth = tensorSize.depth;

        this.scale = scale; // запоминаем коэффициент уменьшения

        mask = new Tensor(inputSize); // создание маски максимумов
    }

    public MaxPoolingLayer(TensorSize tensorSize, int scale) {
        // запоминаем входной размер
        inputSize.width = tensorSize.width;
        inputSize.height = tensorSize.height;
        inputSize.depth = tensorSize.depth;

        // вычисляем выходной размер
        outputSize.width = tensorSize.width / scale;
        outputSize.height = tensorSize.height / scale;
        outputSize.depth = tensorSize.depth;

        this.scale = scale; // запоминаем коэффициент уменьшения

        mask = new Tensor(inputSize); // создание маски максимумов
    }

    // прямое распространение с использованием маски
    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(outputSize); // создаем выходной тензор

        // проходимся по каждому из каналов
        for (int d = 0; d < inputSize.depth; d++) {
            for (int h = 0; h < inputSize.height; h += scale) {
                for (int w = 0; w < inputSize.width; w += scale) {
                    int hMax = h; // индекс строки мвксимума
                    int wMax = w; // индекс столбца максимума
                    BigDecimal max = input.getByIndex(d, h, w); // начальное значение максимума - значение первой клетки подматрицы

                    // проходимся по подматрице и ищем максмум и его координаты
                    for (int i = h; i < h + scale; i++) {
                        for (int j = w; j < w + scale; j++) {
                            BigDecimal value = input.getByIndex(d, i, j); // получаем значение входного тензора
                            mask.setByIndex(d, i, j, BigDecimal.ZERO);

                            // если значение больше максимального
                            if (value.compareTo(max) > 0) {
                                max = value; // обновляем максимум
                                hMax = i; // обновляем индекс строки максимума
                                wMax = j; // обновляем индекс столбца максимума
                            }
                        }
                    }

                    output.setByIndex(d, h / scale, w / scale, max); // записываем в выходной тензор найденый максимум
                    mask.setByIndex(d, hMax, wMax, BigDecimal.ONE); // записываем 1 на место максимального элемента
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    // обратное распространение
    public Tensor backward(final Tensor dOut, final Tensor input) {
        Tensor gradInput = new Tensor(inputSize); // создаем тензор для градиентов

        for (int d = 0; d < inputSize.depth; d++) {
            for (int h = 0; h < inputSize.height; h++) {
                for (int w = 0; w < inputSize.width; w++)
                    gradInput.setByIndex(d, h, w, dOut.getByIndex(d, h/scale, w/scale).multiply(mask.getByIndex(d, h, w), mathContext20)); // умножаем градиенты на маску
            }
        }

        return gradInput; // возвращаем посчитанные градиенты
    }
}























