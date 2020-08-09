import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;

public class NeuralNetwork {

    private static final MathContext mathContext20 = new MathContext(20);

    // гиперпараметры
    private static final BigDecimal epsilon = BigDecimal.valueOf(0.8);
    private static final BigDecimal alpha = BigDecimal.valueOf(0.08);
    private final int classesCount;

    private BigDecimal grad;
    private final BigDecimal[][][] synapticWeights1;
    private final BigDecimal[][][] deltasSynapticWeights1;
    private final BigDecimal[][] synapticWeights2;
    private final BigDecimal[][] deltasSynapticWeights2;

    public NeuralNetwork(BigDecimal[][][] input, BigDecimal[][] output) {

        classesCount = output[0].length; // запись кол-ва классов

        // инициализация весов для входного слоя
        synapticWeights1 = new BigDecimal[classesCount][][];
        deltasSynapticWeights1 = new BigDecimal[classesCount][][];
        for (int c = 0; c < classesCount; c++) { // для каждого класса
            synapticWeights1[c] = new BigDecimal[input[0].length][];
            deltasSynapticWeights1[c] = new BigDecimal[input[0].length][];
            for (int h = 0; h < input[0].length; h++) { // для каждого канала
                synapticWeights1[c][h] = new BigDecimal[input[0][0].length];
                for (int w = 0; w < input[0][0].length; w++)
                    synapticWeights1[c][h][w] = BigDecimal.valueOf(Math.random() - 0.5);
                deltasSynapticWeights1[c][h] = new BigDecimal[input[0][0].length];
                Arrays.fill(deltasSynapticWeights1[c][h], BigDecimal.ZERO);
            }
        }

        // инициализация весов для скрытого слоя
        synapticWeights2 = new BigDecimal[classesCount][];
        deltasSynapticWeights2 = new BigDecimal[classesCount][];
        for (int c = 0; c < classesCount; c++) { // для каждого класса
            synapticWeights2[c] = new BigDecimal[input[0].length + 1];
            for (int h = 0; h < input[0].length + 1; h++)
                synapticWeights2[c][h] = BigDecimal.valueOf(Math.random() - 0.5);
            deltasSynapticWeights2[c] = new BigDecimal[input[0].length + 1];
            Arrays.fill(deltasSynapticWeights2[c], BigDecimal.ZERO);
        }

        training(input, output); // переходим к тренировке/обучению
    }

    public void training(BigDecimal[][][] input, BigDecimal[][] output) {

        BigDecimal[][] output1;
        BigDecimal[] output2;
        BigDecimal[] deltas3;
        BigDecimal[][] deltas2;
        BigDecimal error;
        int statistic;
        boolean statAllTrue;
        int N = 2000; // кол-во эпох
        for (int n = 0; n < N; n++) {
            error = BigDecimal.ZERO;
            statistic = 0;
            for (int i = 0; i < input.length; i++) {
                // прямое распространение
                output1 = forwardFirst(input[i], synapticWeights1);
                // функция активации
                for (int c = 0; c < classesCount; c++) {
                    for (int x = 0; x < output1[c].length; x++)
                        output1[c][x] = BigDecimal.valueOf(sigmoid(output1[c][x].doubleValue()));
                }
                output2 = forward(output1, synapticWeights2);

                // функция активации
                for (int c = 0; c < classesCount; c++)
                    output2[c] = BigDecimal.valueOf(sigmoid(output2[c].doubleValue()));

                // расчет ошибки и статистики
                statAllTrue = true;
                for (int c = 0; c < classesCount; c++) {
                    if ((int) Math.round(output2[c].doubleValue()) != output[i][c].intValue()) {
                        error = error.add((output2[c].subtract(output[i][c], mathContext20)).pow(2, mathContext20));
                        statAllTrue = false;
                    }
                }
                if (statAllTrue)
                    statistic++;

                // обратное распространение
                deltas3 = backwardLast(output2, output[i]);
                deltas2 = backward(output1, synapticWeights2, deltasSynapticWeights2, deltas3);
                backwardFirst(input[i], synapticWeights1, deltasSynapticWeights1, deltas2);
            }

            // вывод ошибки и статистики
            System.out.println(n + " Error: " + error);
            System.out.println(n + " Statistic: " + (double) statistic/input.length);
            System.out.println();

            if (error.equals(BigDecimal.ZERO) || (double) statistic/input.length == 1.) {
                System.out.println("!!! CONGRATULATIONS !!!");
                break;
            }
        }
    }

    // прямое распространение от входного слоя
    public BigDecimal[][] forwardFirst(BigDecimal[][] input, BigDecimal[][][] weights) {
        BigDecimal[][] output = new BigDecimal[classesCount][];
        for (int c = 0; c < classesCount; c++) { // проходимся по классам
            output[c] = new BigDecimal[input.length + 1];
            Arrays.fill(output[c], BigDecimal.ZERO);
            for (int h = 0; h < input.length; h++) { // проходимся по каналам изображения (R, G, B)
                for (int w = 0; w < input[0].length; w++) // проходимся по значениям пикселей канала изображения
                    output[c][h] = output[c][h].add(input[h][w].multiply(weights[c][h][w], mathContext20), mathContext20);
            }
            output[c][input.length] = BigDecimal.ONE; // добавляем смещение
        }
        return output;
    }

    // прямое распространение от скрытого слоя
    public BigDecimal[] forward(BigDecimal[][] input, BigDecimal[][] weights) {
        BigDecimal[] output = new BigDecimal[classesCount];
        for (int c = 0; c < classesCount; c++) { // проходимся по классам
            output[c] = BigDecimal.ZERO;
            for (int h = 0; h < input[0].length; h++) // проходимся по каналам + смещение
                output[c] = output[c].add(input[c][h].multiply(weights[c][h], mathContext20), mathContext20);
        }
        return output;
    }

    // обратное распространение на выходном слое
    public BigDecimal[] backwardLast(BigDecimal[] outActual, BigDecimal[] outIdeal) {
        BigDecimal[] deltas = new BigDecimal[classesCount];
        for (int c = 0; c < classesCount; c++) { // проходимся по классам
            /*deltas[c] = (outIdeal[c].subtract(outActual[c], mathContext20))
                    .multiply(BigDecimal.ONE.subtract(outActual[c]), mathContext20)
                    .multiply(outActual[c], mathContext20);*/
            /*if ( (int) Math.round(outActual[c].doubleValue()) > (int) Math.round(outIdeal[c].doubleValue()) )
                deltas[c] = ((outActual[c].subtract(outIdeal[c], mathContext20)).pow(2, mathContext20))
                        .divide(BigDecimal.valueOf(-classesCount), mathContext20);
            else if ((int) Math.round(outActual[c].doubleValue()) < (int) Math.round(outIdeal[c].doubleValue()))
                deltas[c] = ((outActual[c].subtract(outIdeal[c], mathContext20)).pow(2, mathContext20))
                        .divide(BigDecimal.valueOf(classesCount), mathContext20);
            else
                deltas[c] = BigDecimal.ZERO;*/
            deltas[c] = (outIdeal[c].subtract(outActual[c], mathContext20)).multiply(BigDecimal.valueOf(classesCount*4), mathContext20);
        }
        return deltas;
    }

    // обратное распространение и обновление весов на скрытом слое
    public BigDecimal[][] backward(BigDecimal[][] outActual,
                                     BigDecimal[][] weights,
                                     BigDecimal[][] deltasWeights,
                                     BigDecimal[] nextLayerDeltas) {
        BigDecimal[][] deltas = new BigDecimal[classesCount][];
        for (int c = 0; c < classesCount; c++) {
            deltas[c] = new BigDecimal[outActual[0].length];
            for (int h = 0; h < outActual[0].length; h++) {
                deltas[c][h] = (weights[c][h].multiply(nextLayerDeltas[c], mathContext20));
//                        .multiply(BigDecimal.ONE.subtract(outActual[c][h]), mathContext20)
//                        .multiply(outActual[c][h], mathContext20);
                grad = nextLayerDeltas[c].multiply(outActual[c][h], mathContext20);
                deltasWeights[c][h] = epsilon.multiply(grad, mathContext20)
                        .add(alpha.multiply(deltasWeights[c][h], mathContext20), mathContext20);
                weights[c][h] = weights[c][h].add(deltasWeights[c][h], mathContext20);
            }
        }
        return deltas;
    }

    // обратное распространение и обновление весов на входном слое
    public void backwardFirst(BigDecimal[][] outActual,
                                     BigDecimal[][][] weights,
                                     BigDecimal[][][] deltasWeights,
                                     BigDecimal[][] nextLayerDeltas) {
        for (int c = 0; c < classesCount; c++) { // проходимся по классам
            for (int h = 0; h < outActual.length; h++) { // проходимся по каналам (R, G, B)
                for (int w = 0; w < outActual[0].length; w++) { // проходимся по значениям пикселей изображения
                    grad = nextLayerDeltas[c][h].multiply(outActual[h][w], mathContext20);
                    deltasWeights[c][h][w] = epsilon.multiply(grad, mathContext20)
                            .add(alpha.multiply(deltasWeights[c][h][w], mathContext20), mathContext20);
                    weights[c][h][w] = weights[c][h][w].add(deltasWeights[c][h][w], mathContext20);
                }
            }
        }
    }

    public double sigmoid(double x) {
        return 1. / (1. + Math.exp(-x));
    }

    public void testing(BigDecimal[][][] input, BigDecimal[][] output) {
        BigDecimal[][] output1;
        BigDecimal[] output2;
        BigDecimal error = BigDecimal.ZERO;
        int statistic = 0;
        boolean statAllTrue;
        for (int i = 0; i < input.length; i++) {
            // прямое распространение
            output1 = forwardFirst(input[i], synapticWeights1);
            // функция активации
            for (int c = 0; c < classesCount; c++) {
                for (int x = 0; x < output1[c].length; x++)
                    output1[c][x] = BigDecimal.valueOf(sigmoid(output1[c][x].doubleValue()));
            }
            output2 = forward(output1, synapticWeights2);

            // функция активации
            for (int c = 0; c < classesCount; c++)
                output2[c] = BigDecimal.valueOf(sigmoid(output2[c].doubleValue()));

            // расчет ошибки и статистики
            statAllTrue = true;
            for (int c = 0; c < classesCount; c++) {
                if ((int) Math.round(output2[c].doubleValue()) != output[i][c].intValue()) {
                    error = error.add((output2[c].subtract(output[i][c], mathContext20)).pow(2, mathContext20));
                    statAllTrue = false;
                }
            }
            if (statAllTrue)
                statistic++;
        }

        // вывод ошибки и статистики
        System.out.println("\n\n=== TESTING ===\n");
        System.out.println("Error: " + error);
        System.out.println("Statistic: " + (double) statistic/input.length);
        System.out.println();
    }
}