import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Random;
import java.util.Vector;

public class FullyConnectedLayer {

    private final MathContext mathContext20 = new MathContext(20);

    // тип активационной функции
    private enum ActivationType {
        None, // без активации
        Sigmoid, // сигмоидальная функция
        Tanh, // гиперболический тангенс
        ReLU, // выпрямитель
        LeakyReLU, // выпрямитель с утечкой
        ELU // экспоненциальный выпрямитель
    }

    private TensorSize inputSize = new TensorSize(); // входной размер
    private TensorSize outputSize = new TensorSize(); // выходной размер

    private int inputsCount; // число входных нейронов
    private int outputsCount; // число выходных нейронов

    private ActivationType activationType; // тип активационной функции
    private Tensor gradFuncActivation; // тензор производных функции активации

    private Matrix weightsMatrix; // матрица весовых коэффициентов
    private Matrix gradWeightsMatrix; // матрица градиентов весовых коэффициентов

    private Vector<BigDecimal> biases; // смещения
    private Vector<BigDecimal> gradBiases; // градиенты смещений

    // получаем активационную функцию
    private ActivationType getActivationType(final String activationType) {
        if (activationType.equals("sigmoid"))
            return ActivationType.Sigmoid;

        if (activationType.equals("tanh"))
            return ActivationType.Tanh;

        if (activationType.equals("relu"))
            return ActivationType.ReLU;

        if (activationType.equals("leakyrelu"))
            return ActivationType.LeakyReLU;

        if (activationType.equals("elu"))
            return ActivationType.ELU;

        if (activationType.equals("none") || activationType.equals(""))
            return ActivationType.None;

        throw new RuntimeException("Invalid activation function");
    }

    // инициализация весовых коэффициентов
    private void initWeights() {
        for (int i = 0; i < outputsCount; i++) {
            for (int j = 0; j < inputsCount; j++)
                weightsMatrix.setByIndex(i, j, BigDecimal.valueOf(new Random().nextGaussian()*Math.sqrt(2. / (inputSize.height*inputSize.width*inputSize.depth))));

            biases.add(BigDecimal.valueOf(0.01)); // все смещения делаем равными 0.01
//            biases.set(i, BigDecimal.valueOf(0.01)); // все смещения делаем равными 0.01
        }
    }

    // применение активационной функции с вычислением значений ее производной
    private void activate(Tensor output) {
        if (activationType == ActivationType.None) {
            for (int i = 0; i < outputsCount; i++)
                gradFuncActivation.setByIndex(i, BigDecimal.ONE);
        }
        else if (activationType == ActivationType.Sigmoid) {
            for (int i = 0; i < outputsCount; i++) {
                output.setByIndex(i, BigDecimal.valueOf(1./(1. + Math.exp(output.getByIndex(i).multiply(BigDecimal.valueOf(-1), mathContext20).doubleValue()))));
                gradFuncActivation.setByIndex(i, output.getByIndex(i).multiply(BigDecimal.ONE.subtract(output.getByIndex(i), mathContext20), mathContext20));
            }
        }
        else if (activationType == ActivationType.Tanh) {
            for (int i = 0; i < outputsCount; i++) {
                output.setByIndex(i, BigDecimal.valueOf(Math.tanh(output.getByIndex(i).doubleValue())));
                gradFuncActivation.setByIndex(i, BigDecimal.ONE.subtract(output.getByIndex(i).multiply(output.getByIndex(i), mathContext20), mathContext20));
            }
        }
        else if (activationType == ActivationType.ReLU) {
            for (int i = 0; i < outputsCount; i++) {
                if (output.getByIndex(i).compareTo(BigDecimal.ZERO) > 0)
                    gradFuncActivation.setByIndex(i, BigDecimal.ONE);
                else {
                    output.setByIndex(i, BigDecimal.ZERO);
                    gradFuncActivation.setByIndex(i, BigDecimal.ZERO);
                }
            }
        }
        else if (activationType == ActivationType.LeakyReLU) {
            for (int i = 0; i < outputsCount; i++) {
                if (output.getByIndex(i).compareTo(BigDecimal.ZERO) > 0)
                    gradFuncActivation.setByIndex(i, BigDecimal.ONE);
                else {
                    output.setByIndex(i, output.getByIndex(i).multiply(BigDecimal.valueOf(0.01), mathContext20));
                    gradFuncActivation.setByIndex(i, BigDecimal.valueOf(0.01));
                }
            }
        }
        else if (activationType == ActivationType.ELU) {
            for (int i = 0; i < outputsCount; i++) {
                if (output.getByIndex(i).compareTo(BigDecimal.ZERO) > 0)
                    gradFuncActivation.setByIndex(i, BigDecimal.ONE);
                else {
                    output.setByIndex(i, BigDecimal.valueOf(Math.exp(output.getByIndex(i).doubleValue())).subtract(BigDecimal.ONE, mathContext20));
                    gradFuncActivation.setByIndex(i, output.getByIndex(i).add(BigDecimal.ONE, mathContext20));
                }
            }
        }
    }

    // создание слоя
    public FullyConnectedLayer(TensorSize size, int outputsCount, final String activationType) {
        weightsMatrix = new Matrix(outputsCount, size.height* size.width*size.depth);
        gradWeightsMatrix = new Matrix(outputsCount, size.height* size.width*size.depth);
        gradFuncActivation = new Tensor(1, 1, outputsCount);

        // запоминаем входной размер
        inputSize.width = size.width;
        inputSize.height = size.height;
        inputSize.depth = size.depth;

        // вычисляем выходной размер
        outputSize.width = 1;
        outputSize.height = 1;
        outputSize.depth = outputsCount;

        this.inputsCount = size.height * size.width * size.depth; // запоминаем число входных нейронов
        this.outputsCount = outputsCount; // запоминаем число выходных нейронов

        this.activationType = getActivationType(activationType); // получаем активационную функцию

        biases = new Vector<>(outputsCount); // создаем вектор смещений
        gradBiases = new Vector<>(outputsCount); // создаём вектор градиентов по весам смещения

        initWeights(); // инициализируем весовые коэффициенты
    }

    public FullyConnectedLayer(TensorSize size, int outputsCount) {
        final String activationType = "none";
        weightsMatrix = new Matrix(outputsCount, size.height* size.width*size.depth);
        gradWeightsMatrix = new Matrix(outputsCount, size.height* size.width*size.depth);
        gradFuncActivation = new Tensor(1, 1, outputsCount);

        // запоминаем входной размер
        inputSize.width = size.width;
        inputSize.height = size.height;
        inputSize.depth = size.depth;

        // вычисляем выходной размер
        outputSize.width = 1;
        outputSize.height = 1;
        outputSize.depth = outputsCount;

        this.inputsCount = size.height * size.width * size.depth; // запоминаем число входных нейронов
        this.outputsCount = outputsCount; // запоминаем число выходных нейронов

        this.activationType = getActivationType(activationType); // получаем активационную функцию

        biases = new Vector<>(outputsCount); // создаем вектор смещений
        gradBiases = new Vector<>(outputsCount); // создаём вектор градиентов по весам смещения

        initWeights(); // инициализируем весовые коэффициенты
    }

    // прямое распространение
    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(outputSize); // создаем выходной вектор

        // проходимся по каждому выходному нейрону
        for (int i = 0; i < outputsCount; i++) {
            BigDecimal sum = biases.get(i); // прибавляем смещение

            // умножаем входной тензор на матрицу
            for (int j = 0; j < inputsCount; j++)
                sum = sum.add(weightsMatrix.getByIndex(i, j).multiply(input.getByIndex(j), mathContext20), mathContext20);

            output.setByIndex(i, sum);
        }

        activate(output); // применяем активационную функцию

        return output; // возвращаем выходной тензор
    }

    // обратное распространение
    public Tensor backward(final Tensor dOut, final Tensor input) {
        // домножаем производные на градиенты следующего слоя для сокращения количества умножений
        for (int i = 0; i < outputsCount; i++)
            gradFuncActivation.setByIndex(i, gradFuncActivation.getByIndex(i).multiply(dOut.getByIndex(i), mathContext20));

        // вычисляем градиенты по весовым коэффициентам
        for (int i = 0; i < outputsCount; i++) {
            for (int j = 0; j < inputsCount; j++)
                gradWeightsMatrix.setByIndex(i, j, gradFuncActivation.getByIndex(i).multiply(input.getByIndex(j), mathContext20));

            gradBiases.add(gradFuncActivation.getByIndex(i));
//            gradBiases.set(i, gradFuncActivation.getByIndex(i));
        }

        Tensor gradInput = new Tensor(inputSize); // создаём тензор для градиентов по входам

        // вычисляем градиенты по входам
        for (int j = 0; j < inputsCount; j++) {
            BigDecimal sum = BigDecimal.ZERO;

            for (int i = 0; i < outputsCount; i++)
                sum = sum.add(weightsMatrix.getByIndex(i, j).multiply(gradFuncActivation.getByIndex(i), mathContext20), mathContext20);

            gradInput.setByIndex(j, sum); // записываем результат в тензор градиентов
        }

        return gradInput; // возвращаем тензор градиентов
    }

    // обновление весовых коэффициентов
    public void updateWeights(BigDecimal learningRate) {
        for (int i = 0; i < outputsCount; i++) {
            for (int j = 0; j < inputsCount; j++)
                weightsMatrix.setByIndex(i, j, weightsMatrix.getByIndex(i, j).subtract(learningRate.multiply(gradWeightsMatrix.getByIndex(i, j), mathContext20), mathContext20));

            biases.set(i, biases.get(i).subtract(learningRate.multiply(gradBiases.get(i), mathContext20), mathContext20)); // обновляем веса смещения
        }
    }

/*    // установка веса матрицы
    public void setWeight(int i, int j, BigDecimal weight) {
        weightsMatrix.setByIndex(i, j, weight);
    }

    // установка веса смещения
    public void setBias(int i, BigDecimal bias) {
        biases.set(i, bias);
    }

    public static void main(String[] args) {
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new TensorSize(1, 1, 8), 4, "relu");

        fullyConnectedLayer.setWeight(0, 0, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(0, 1, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(0, 2, BigDecimal.valueOf(3));
        fullyConnectedLayer.setWeight(0, 3, BigDecimal.valueOf(4));
        fullyConnectedLayer.setWeight(0, 4, BigDecimal.valueOf(-1));
        fullyConnectedLayer.setWeight(0, 5, BigDecimal.valueOf(-2));
        fullyConnectedLayer.setWeight(0, 6, BigDecimal.valueOf(-3));
        fullyConnectedLayer.setWeight(0, 7, BigDecimal.valueOf(-4));

        fullyConnectedLayer.setWeight(1, 0, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(1, 1, BigDecimal.valueOf(0));
        fullyConnectedLayer.setWeight(1, 2, BigDecimal.valueOf(0));
        fullyConnectedLayer.setWeight(1, 3, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(1, 4, BigDecimal.valueOf(-1));
        fullyConnectedLayer.setWeight(1, 5, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(1, 6, BigDecimal.valueOf(-3));
        fullyConnectedLayer.setWeight(1, 7, BigDecimal.valueOf(4));

        fullyConnectedLayer.setWeight(2, 0, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(2, 1, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(2, 2, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(2, 3, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(2, 4, BigDecimal.valueOf(-3));
        fullyConnectedLayer.setWeight(2, 5, BigDecimal.valueOf(-4));
        fullyConnectedLayer.setWeight(2, 6, BigDecimal.valueOf(-5));
        fullyConnectedLayer.setWeight(2, 7, BigDecimal.valueOf(-6));

        fullyConnectedLayer.setWeight(3, 0, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(3, 1, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(3, 2, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(3, 3, BigDecimal.valueOf(1));
        fullyConnectedLayer.setWeight(3, 4, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(3, 5, BigDecimal.valueOf(2));
        fullyConnectedLayer.setWeight(3, 6, BigDecimal.valueOf(-3));
        fullyConnectedLayer.setWeight(3, 7, BigDecimal.valueOf(-8));

        fullyConnectedLayer.setBias(0, BigDecimal.ZERO);
        fullyConnectedLayer.setBias(1, BigDecimal.ZERO);
        fullyConnectedLayer.setBias(2, BigDecimal.ZERO);
        fullyConnectedLayer.setBias(3, BigDecimal.ZERO);

        Tensor input = new Tensor(8, 1, 1);
        input.setByIndex(0, 0, 0, BigDecimal.ONE);
        input.setByIndex(0, 0, 1, BigDecimal.valueOf(2));
        input.setByIndex(0, 0, 2, BigDecimal.valueOf(-3));
        input.setByIndex(0, 0, 3, BigDecimal.valueOf(4));
        input.setByIndex(0, 0, 4, BigDecimal.ZERO);
        input.setByIndex(0, 0, 5, BigDecimal.valueOf(-7));
        input.setByIndex(0, 0, 6, BigDecimal.valueOf(2));
        input.setByIndex(0, 0, 7, BigDecimal.valueOf(-4));

        fullyConnectedLayer.forward(input).print();

        Tensor dout = new Tensor(4, 1, 1);
        dout.setByIndex(0, 0, 0, BigDecimal.valueOf(-0.5));
        dout.setByIndex(0, 0, 1, BigDecimal.valueOf(0.1));
        dout.setByIndex(0, 0, 2, BigDecimal.valueOf(-0.25));
        dout.setByIndex(0, 0, 3, BigDecimal.valueOf(0.7));

        fullyConnectedLayer.backward(dout, input).print();
    }*/
}
