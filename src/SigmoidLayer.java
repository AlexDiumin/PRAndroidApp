import java.math.BigDecimal;
import java.math.MathContext;

public class SigmoidLayer {

    private final MathContext mathContext20 = new MathContext(20);

    private TensorSize size = new TensorSize(); // размер слоя

    // создание слоя
    public SigmoidLayer(TensorSize size) {
        this.size = size; // сохраняем размер
    }

    // активационная функция - сигмоида
    public BigDecimal sigmoid(BigDecimal x) {
        return BigDecimal.valueOf(1./(1. + Math.exp(x.multiply(BigDecimal.valueOf(-1), mathContext20).doubleValue())));
    }

    // прямое распространение
    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(size); // создаем выходной тензор

        // проходимся по всем значениям входного тензора
        for (int h = 0; h < size.height; h++) {
            for (int w = 0; w < size.width; w++) {
                for (int d = 0; d < size.depth; d++) {
                    /*// +++ -----------------------------------------
                    // вычисляем значение функции активации
                    if (input.getByIndex(d, h, w).compareTo(BigDecimal.ZERO) > 0)
                        output.setByIndex(d, h, w, input.getByIndex(d, h, w));
                    else
                        output.setByIndex(d, h, w, BigDecimal.ZERO);
                    // ---------------------------------------------*/
                    output.setByIndex(d, h, w, sigmoid(input.getByIndex(d, h, w))); // вычисляем значение функции активации
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    // обратное распространение
    public Tensor backward(final Tensor dOut, final Tensor input) {
        Tensor gradInput = new Tensor(size); // создаем тензор градиентов

        // проходимся по всем значениям тензора градиентов
        BigDecimal sigmoidResult;
        for (int h = 0; h < size.height; h++) {
            for (int w = 0; w < size.width; w++) {
                for (int d = 0; d < size.depth; d++) {
                    /*// +++ -----------------------------------------
                    if (input.getByIndex(d, h, w).compareTo(BigDecimal.ZERO) > 0)
                        gradInput.setByIndex(d, h, w, dOut.getByIndex(d, h, w));
                    else
                        gradInput.setByIndex(d, h, w, BigDecimal.ZERO);
                    // ---------------------------------------------*/
                    sigmoidResult = sigmoid(input.getByIndex(d, h, w));
                    // умножаем градиент следующего слоя на производную функции активации
                    gradInput.setByIndex(d, h, w, dOut.getByIndex(d, h, w).multiply(sigmoidResult.multiply(BigDecimal.ONE.subtract(sigmoidResult, mathContext20), mathContext20), mathContext20));
                }
            }
        }

        return gradInput; // возвращаем тензор градиентов
    }
}
























