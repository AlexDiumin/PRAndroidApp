import java.math.BigDecimal;
import java.math.MathContext;

public class FullyNected {

    private static final MathContext mathContext20 = new MathContext(20);

    private final TensorSize inputSize; // размер входного тензора
    private final TensorSize outputSize; // размер выходного тензора

    private final Tensor[][][] weights; // веса
    private final Tensor biases; // смещения

    private final Tensor[][][] gradWeights; // градиенты весов
    private final Tensor gradBiases; // градиенты смещений

    private final Tensor[][][] deltaWeights; // разница между старыми и новыми значениями весов
    private final Tensor deltaBiases; // разница между старыми и новыми значениями смещений

    private final Tensor derivativesReLU; // тензор производных ReLU

    public FullyNected(final TensorSize inputSize, final TensorSize outputSize) {
        this.inputSize = inputSize; // запоминаем входной размер тензора
        this.outputSize = outputSize; // запоминаем выходной размер тензора

        this.weights = new Tensor[this.outputSize.depth][][];
        this.biases = new Tensor(this.outputSize);
        this.gradWeights = new Tensor[this.outputSize.depth][][];
        this.gradBiases = new Tensor(this.outputSize); // инициализация градиентов смещений нулями
        this.deltaWeights = new Tensor[this.outputSize.depth][][];
        this.deltaBiases = new Tensor(this.outputSize); // инициализация разниц смещений нулями
        double variance = 1. / (this.inputSize.depth * this.inputSize.height * this.inputSize.width); // дисперсия
        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            this.weights[oD] = new Tensor[this.outputSize.height][];
            this.gradWeights[oD] = new Tensor[this.outputSize.height][];
            this.deltaWeights[oD] = new Tensor[this.outputSize.height][];
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                this.weights[oD][oH] = new Tensor[this.outputSize.width];
                this.gradWeights[oD][oH] = new Tensor[this.outputSize.width];
                this.deltaWeights[oD][oH] = new Tensor[this.outputSize.width];
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    this.weights[oD][oH][oW] = new Tensor(this.inputSize);
                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++)
                                this.weights[oD][oH][oW].setByIndex(iD, iH, iW, BigDecimal.valueOf(variance*Math.random())); // инициализируем веса
                        }
                    }
                    this.biases.setByIndex(oD, oH, oW, BigDecimal.valueOf(variance*Math.random())); // инициализируем веса
                    this.gradWeights[oD][oH][oW] = new Tensor(this.inputSize); // инициализация градиентов весов нулями
                    this.deltaWeights[oD][oH][oW] = new Tensor(this.inputSize); // инициализация разниц весов нулями
                }
            }
        }

        this.derivativesReLU = new Tensor(this.outputSize); // создаем тензор производных ReLU
    }

    // для последнего слоя
    public FullyNected(final TensorSize inputSize, final TensorSize outputSize, boolean lastLayer) {
        this.inputSize = inputSize; // запоминаем входной размер тензора
        this.outputSize = outputSize; // запоминаем выходной размер тензора

        this.weights = new Tensor[this.outputSize.depth][][];
        this.biases = new Tensor(this.outputSize);
        this.gradWeights = new Tensor[this.outputSize.depth][][];
        this.gradBiases = new Tensor(this.outputSize); // инициализация градиентов смещений нулями
        this.deltaWeights = new Tensor[this.outputSize.depth][][];
        this.deltaBiases = new Tensor(this.outputSize); // инициализация разниц смещений нулями
        double variance = 2. / (this.inputSize.depth * this.inputSize.height * this.inputSize.width + this.outputSize.depth * this.outputSize.height * this.outputSize.width); // дисперсия

        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            this.weights[oD] = new Tensor[this.outputSize.height][];
            this.gradWeights[oD] = new Tensor[this.outputSize.height][];
            this.deltaWeights[oD] = new Tensor[this.outputSize.height][];
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                this.weights[oD][oH] = new Tensor[this.outputSize.width];
                this.gradWeights[oD][oH] = new Tensor[this.outputSize.width];
                this.deltaWeights[oD][oH] = new Tensor[this.outputSize.width];
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    this.weights[oD][oH][oW] = new Tensor(this.inputSize);
                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++)
                                this.weights[oD][oH][oW].setByIndex(iD, iH, iW, BigDecimal.valueOf(2.*variance*Math.random() - variance)); // инициализируем веса
                        }
                    }
                    this.biases.setByIndex(oD, oH, oW, BigDecimal.valueOf(2.*variance*Math.random() - variance)); // инициализируем веса
                    this.gradWeights[oD][oH][oW] = new Tensor(this.inputSize); // инициализация градиентов весов нулями
                    this.deltaWeights[oD][oH][oW] = new Tensor(this.inputSize); // инициализация разниц весов нулями
                }
            }
        }

        this.derivativesReLU = null; // sigmoid не нуждаеться в запоминаниии производных
    }

    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(this.outputSize); // создаем выходной тензор

        BigDecimal sum;
        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    sum = this.biases.getByIndex(oD, oH, oW); // сразу прибавляем смещение

                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++)
                                sum = sum.add(input.getByIndex(iD, iH, iW)
                                        .multiply(this.weights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20), mathContext20);
                        }
                    }

                    // вычисляем и запоминаем производные ReLU
                    this.derivativesReLU.setByIndex(oD, oH, oW,
                            (sum.compareTo(BigDecimal.ZERO) > 0 ? BigDecimal.ONE : BigDecimal.valueOf(0.01)));

                    sum = sum.compareTo(BigDecimal.ZERO) >= 0 ? sum : sum.multiply(BigDecimal.valueOf(0.01), mathContext20); // ReLU

                    output.setByIndex(oD, oH, oW, sum); // записываем сумму в выходной тензор
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    // для последнего слоя
    public Tensor forward(final Tensor input, boolean lastLayer) {
        Tensor output = new Tensor(this.outputSize); // создаем выходной тензор

        BigDecimal sum;
        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    sum = this.biases.getByIndex(oD, oH, oW); // сразу прибавляем смещение

                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++)
                                sum = sum.add(input.getByIndex(iD, iH, iW)
                                        .multiply(this.weights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20), mathContext20);
                        }
                    }

                    sum = this.sigmoid(sum); // sigmoid

                    output.setByIndex(oD, oH, oW, sum); // записываем сумму в выходной тензор
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    public Tensor backward(final Tensor deltasOut, final Tensor input) {

        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {

                    // домножаем выходные дельты на производные ReLU
                    deltasOut.setByIndex(oD, oH, oW, deltasOut.getByIndex(oD, oH, oW)
                            .multiply(this.derivativesReLU.getByIndex(oD, oH, oW), mathContext20));

                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++) {
                                // считаем градиенты по весам для текущего слоя
                                this.gradWeights[oD][oH][oW].setByIndex(iD, iH, iW, deltasOut.getByIndex(oD, oH, oW)
                                        .multiply(input.getByIndex(iD, iH,iW), mathContext20));
                            }
                        }
                    }

                    // записываем градиенты по смещениям текущего слоя
                    this.gradBiases.setByIndex(oD, oH, oW, deltasOut.getByIndex(oD, oH, oW));
                }
            }
        }

        Tensor deltasIn = new Tensor(this.inputSize); // создаем тензор дельт для предыдущего слоя
        BigDecimal sum;
        // проходимся по входному тензору
        for (int iD = 0; iD < this.inputSize.depth; iD++) {
            for (int iH = 0; iH < this.inputSize.height; iH++) {
                for (int iW = 0; iW < this.inputSize.width; iW++) {
                    sum = BigDecimal.ZERO;

                    // проходимся по выходному тензору
                    for (int oD = 0; oD < this.outputSize.depth; oD++) {
                        for (int oH = 0; oH < this.outputSize.height; oH++) {
                            for (int oW = 0; oW < this.outputSize.width; oW++) {
                                // наращиваем сумму
                                sum = sum.add(deltasOut.getByIndex(oD, oH, oW)
                                        .multiply(this.weights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20), mathContext20);
                            }
                        }
                    }

                    deltasIn.setByIndex(iD, iH, iW, sum); // записываем сумму в выходной тензор
                }
            }
        }
        return deltasIn; // возвращаем тензор дельт для предыдущего слоя
    }

    // для последнего слоя
    public Tensor backward(final Tensor deltasOut, final Tensor input, final Tensor output) {

        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {

                    // домножаем выходные дельты на производные sigmoid
                    deltasOut.setByIndex(oD, oH, oW, deltasOut.getByIndex(oD, oH, oW)
                            .multiply((BigDecimal.ONE.subtract(output.getByIndex(oD, oH, oW), mathContext20))
                                    .multiply(output.getByIndex(oD, oH, oW), mathContext20), mathContext20));

                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++) {
                                // считаем градиенты по весам для текущего слоя
                                this.gradWeights[oD][oH][oW].setByIndex(iD, iH, iW, deltasOut.getByIndex(oD, oH, oW)
                                        .multiply(input.getByIndex(iD, iH,iW), mathContext20));
                            }
                        }
                    }

                    // записываем градиенты по смещениям текущего слоя
                    this.gradBiases.setByIndex(oD, oH, oW, deltasOut.getByIndex(oD, oH, oW));
                }
            }
        }

        Tensor deltasIn = new Tensor(this.inputSize); // создаем тензор дельт для предыдущего слоя
        BigDecimal sum;
        // проходимся по входному тензору
        for (int iD = 0; iD < this.inputSize.depth; iD++) {
            for (int iH = 0; iH < this.inputSize.height; iH++) {
                for (int iW = 0; iW < this.inputSize.width; iW++) {
                    sum = BigDecimal.ZERO;

                    // проходимся по выходному тензору
                    for (int oD = 0; oD < this.outputSize.depth; oD++) {
                        for (int oH = 0; oH < this.outputSize.height; oH++) {
                            for (int oW = 0; oW < this.outputSize.width; oW++) {
                                // наращиваем сумму
                                sum = sum.add(deltasOut.getByIndex(oD, oH, oW)
                                        .multiply(this.weights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20), mathContext20);
                            }
                        }
                    }

                    deltasIn.setByIndex(iD, iH, iW, sum); // записываем сумму в выходной тензор
                }
            }
        }
        return deltasIn; // возвращаем тензор дельт для предыдущего слоя
    }

    public void updateWeights(BigDecimal epsilon, BigDecimal alpha) {
        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    // проходимся по входному тензору
                    for (int iD = 0; iD < this.inputSize.depth; iD++) {
                        for (int iH = 0; iH < this.inputSize.height; iH++) {
                            for (int iW = 0; iW < this.inputSize.width; iW++) {
                                // считаем новую разницу весов
                                this.deltaWeights[oD][oH][oW].setByIndex(iD, iH, iW,
                                        (epsilon.multiply(this.gradWeights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20))
                                                .add(alpha.multiply(this.deltaWeights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20), mathContext20));

                                // обновляем веса
                                this.weights[oD][oH][oW].setByIndex(iD, iH, iW, this.weights[oD][oH][oW].getByIndex(iD, iH, iW)
                                        .add(this.deltaWeights[oD][oH][oW].getByIndex(iD, iH, iW), mathContext20));

                                this.gradWeights[oD][oH][oW].setByIndex(iD, iH, iW, BigDecimal.ZERO); // обнуляем градиенты весов
                            }
                        }
                    }

                    // считаем новую разницу смещений
                    this.deltaBiases.setByIndex(oD, oH, oW,
                            (epsilon.multiply(this.gradBiases.getByIndex(oD, oH, oW), mathContext20))
                                    .add(alpha.multiply(this.deltaBiases.getByIndex(oD, oH, oW), mathContext20), mathContext20));

                    // обновляем смещения
                    this.biases.setByIndex(oD, oH, oW, this.biases.getByIndex(oD, oH, oW)
                            .add(this.deltaBiases.getByIndex(oD, oH, oW), mathContext20));

                    this.gradBiases.setByIndex(oD, oH, oW, BigDecimal.ZERO); // обнуляем градиенты смещений
                }
            }
        }
    }

    public final TensorSize getOutputSize() {
        return this.outputSize;
    }

    public BigDecimal sigmoid(BigDecimal x) {
        return BigDecimal.valueOf(1. / (1. + Math.pow(Math.E, -x.doubleValue())));
    }
}
