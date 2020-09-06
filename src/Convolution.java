import java.math.BigDecimal;
import java.math.MathContext;

public class Convolution {

    private static final MathContext mathContext20 = new MathContext(20);

    private final TensorSize inputSize; // размер входного тензора
    private final TensorSize outputSize; // размер выходного тензора

    private final int filtersCount; // кол-во фильтров
    private final int filterSize; // размер фильтров (= 3: 3х3)
    private final int stride; // шаг свертки (= 1)
    private final int padding; // дополнение нулями (= 1)

    private final Tensor[] filters; // фильтры
    private final BigDecimal[] biases; // смещение

    private final Tensor[] gradFilters; // градиенты значений фильтров
    private final BigDecimal[] gradBiases; // градиенты смещений

    private final Tensor[] deltaFiltersWeights; // разница между старыми и новыми значениями весов фильтров
    private final BigDecimal[] deltaBiasesWeights; // разница между старыми и новыми значениями весов смещений

    private final Tensor derivativesReLU; // тензор производных ReLU

    public Convolution(TensorSize inputSize, int filterSize, int filtersCount, int stride, int padding) {
        this.inputSize = inputSize; // запоминаем размер входного тензора
        // запоминаем размер выходного тензора
        this.outputSize = new TensorSize(filtersCount,
                (this.inputSize.height - filterSize + 2*padding)/stride + 1,
                (this.inputSize.width - filterSize + 2*padding)/stride + 1);

        this.filtersCount = filtersCount;
        this.filterSize = filterSize;
        this.stride = stride;
        this.padding = padding;

        // инициализация фильтров, смещений, градиентов фильтров и смещений, разниц весов фильтров
        this.filters = new Tensor[filtersCount];
        this.biases = new BigDecimal[filtersCount];
        this.gradFilters = new Tensor[filtersCount];
        this.gradBiases = new BigDecimal[filtersCount];
        this.deltaFiltersWeights = new Tensor[filtersCount];
        this.deltaBiasesWeights = new BigDecimal[filtersCount];
        double variance = 1. / (this.inputSize.depth * this.inputSize.height * this.inputSize.width); // дисперсия
        for (int f = 0; f < this.filtersCount; f++) {
            this.filters[f] = new Tensor(this.inputSize.depth, this.filterSize, this.filterSize);
            for (int d = 0; d < this.inputSize.depth; d++) {
                for (int h = 0; h < this.filterSize; h++) {
                    for (int w = 0; w < this.filterSize; w++)
                        this.filters[f].setByIndex(d, h, w, BigDecimal.valueOf(variance*Math.random()));
//                        this.filters[f].setByIndex(d, h, w, BigDecimal.valueOf(2.*variance*Math.random() - variance));
                }
            }
            this.biases[f] = BigDecimal.valueOf(variance*Math.random()); // инициализация смещений
//            this.biases[f] = BigDecimal.valueOf(2.*variance*Math.random() - variance); // инициализация смещений
            this.gradFilters[f] = new Tensor(this.inputSize.depth, this.filterSize, this.filterSize);
            this.gradBiases[f] = BigDecimal.ZERO; // инициализация градиентов смещений
            this.deltaFiltersWeights[f] = new Tensor(this.inputSize.depth, this.filterSize, this.filterSize);
            this.deltaBiasesWeights[f] = BigDecimal.ZERO;
        }

        this.derivativesReLU = new Tensor(this.outputSize); // создаем тензор производных ReLU
    }

    public Tensor forward(final Tensor input) {
        // созданее выходного тензора
        Tensor output = new Tensor(this.outputSize);

        BigDecimal sum; // сумма на одном шаге одного фильтра
        int H, W; // индексы с учетом размеров шага и отступа
        // проходимся по фильтрам
        for (int f = 0; f < this.filtersCount; f++) {
            // проходимся по выходному тензору
            for (int h = 0; h < this.outputSize.height; h++) {
                for (int w = 0; w < this.outputSize.width; w++) {
                    sum = this.biases[f]; // сразу прибавляем смещение

                    // проходимся по фильтру
                    for (int fH = 0; fH < this.filterSize; fH++) {
                        for (int fW = 0; fW < this.filterSize; fW++) {
                            H = this.stride*h + fH - this.padding;
                            W = this.stride*w + fW - this.padding;

                            // поскольку вне границ входного тензора элементы нулевые, то игнорируем их
                            if (H < 0 || H >= this.inputSize.height || W < 0 || W >= this.inputSize.width)
                                continue;

                            // проходимся по глубине тензора и считаем сумму
                            for (int d = 0; d < this.filters[f].getDepth(); d++)
                                sum = sum.add(input.getByIndex(d, H, W)
                                        .multiply(this.filters[f].getByIndex(d, fH, fW), mathContext20), mathContext20);
                        }
                    }

                    /*// вычисляем и запоминаем производные ReLU
                    this.derivativesReLU.setByIndex(f, h, w,
                            (sum.compareTo(BigDecimal.ZERO) > 0 ? BigDecimal.ONE : BigDecimal.ZERO));

                    sum = sum.compareTo(BigDecimal.ZERO) >= 0 ? sum : BigDecimal.ZERO; // ReLU*/
                    // вычисляем и запоминаем производные ReLU
                    this.derivativesReLU.setByIndex(f, h, w,
                            (sum.compareTo(BigDecimal.ZERO) > 0 ? BigDecimal.ONE : BigDecimal.valueOf(0.01)));
                    sum = sum.compareTo(BigDecimal.ZERO) >= 0 ? sum : sum.multiply(BigDecimal.valueOf(0.01), mathContext20); // ReLU



                    output.setByIndex(f, h, w, sum); // записываем результат свертки в выходной тензор
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
                }
            }
        }

        BigDecimal sum; // для подсчета сум градиентов и дельт
        int H, W; // индексы с учетом отступов

        // проходимся по фильтрам
        for (int f = 0; f < this.filtersCount; f++) {
            for (int d = 0; d < this.filters[f].getDepth(); d++) {
                for (int h = 0; h < this.filterSize; h++) {
                    for (int w = 0; w < this.filterSize; w++) {
                        sum = BigDecimal.ZERO;

                        for (int dH = 0; dH < deltasOut.getHeight(); dH++) {
                            for (int dW = 0; dW < deltasOut.getWidth(); dW++) {
                                H = this.stride*h + dH - this.padding;
                                W = this.stride*w + dW - this.padding;

                                // поскольку вне границ входного тензора элементы нулевые, то игнорируем их
                                if (H < 0 || H >= this.inputSize.height || W < 0 || W >= this.inputSize.width)
                                    continue;

                                // наращиваем градиенты весов фильтров
                                sum = sum.add(input.getByIndex(d, H, W)
                                        .multiply(deltasOut.getByIndex(f, dH, dW), mathContext20), mathContext20);
                            }
                        }

                        gradFilters[f].setByIndex(d, h, w, sum); // записываем значения градиента
                    }
                }
            }

            // наращиваем градиенты смещения
            for (int h = 0; h < deltasOut.getHeight(); h++) {
                for (int w = 0; w < deltasOut.getWidth(); w++)
                    gradBiases[f] = gradBiases[f].add(deltasOut.getByIndex(f, h, w), mathContext20);
            }
        }

        Tensor deltasIn = new Tensor(this.inputSize); // создаем тензор дельт для предыдущего слоя
        // расчитываем значения дельт
//        for (int d = 0; d < this.filters[0].getDepth(); d++) {
        for (int d = 0; d < this.inputSize.depth; d++) {
            for (int h = 0; h < this.inputSize.height; h++) {
                for (int w = 0; w < this.inputSize.width; w++) {
                    sum = BigDecimal.ZERO;

                    // обратная свертка с перевернутой матрицей фильтров
                    for (int fH = this.filterSize - 1; fH >= 0; fH--) {
                        for (int fW = this.filterSize - 1; fW >= 0; fW--) {
                            H = h + fH - this.padding;
                            W = w + fW - this.padding;

                            // поскольку вне границ тензора элементы нулевые, то игнорируем их
                            if (H < 0 || H >= deltasOut.getHeight() || W < 0 || W >= deltasOut.getWidth())
                                continue;

                            // суммируем по всем фильтрам
                            for (int f = 0; f < this.filtersCount; f++)
                                sum = sum.add(deltasOut.getByIndex(f, H, W)
                                        .multiply(this.filters[f].getByIndex(d, fH, fW), mathContext20), mathContext20);
                                /*sum = sum.add(deltasOut.getByIndex(d, H, W)
                                        .multiply(this.filters[f].getByIndex(d, fH, fW), mathContext20), mathContext20);*/
                        }
                    }

                    deltasIn.setByIndex(d, h, w, sum); // записываем результат в тензор дельт по входам
                }
            }
        }
        return deltasIn;
    }

    // для самого первого слоя
    public void backward(final Tensor deltasOut, final Tensor input, boolean firstLayer) {
        // проходимся по выходному тензору
        for (int oD = 0; oD < this.outputSize.depth; oD++) {
            for (int oH = 0; oH < this.outputSize.height; oH++) {
                for (int oW = 0; oW < this.outputSize.width; oW++) {
                    // домножаем выходные дельты на производные ReLU
                    deltasOut.setByIndex(oD, oH, oW, deltasOut.getByIndex(oD, oH, oW)
                            .multiply(this.derivativesReLU.getByIndex(oD, oH, oW), mathContext20));
                }
            }
        }

        BigDecimal sum; // для подсчета сум градиентов и дельт
        int H, W; // индексы с учетом отступов

        // проходимся по фильтрам
        for (int f = 0; f < this.filtersCount; f++) {
            for (int d = 0; d < this.filters[f].getDepth(); d++) {
                for (int h = 0; h < this.filterSize; h++) {
                    for (int w = 0; w < this.filterSize; w++) {
                        sum = BigDecimal.ZERO;

                        for (int dH = 0; dH < deltasOut.getHeight(); dH++) {
                            for (int dW = 0; dW < deltasOut.getWidth(); dW++) {
                                H = this.stride*h + dH - this.padding;
                                W = this.stride*w + dW - this.padding;

                                // поскольку вне границ входного тензора элементы нулевые, то игнорируем их
                                if (H < 0 || H >= this.inputSize.height || W < 0 || W >= this.inputSize.width)
                                    continue;

                                // наращиваем градиенты весов фильтров
                                sum = sum.add(input.getByIndex(d, H, W)
                                        .multiply(deltasOut.getByIndex(f, dH, dW), mathContext20), mathContext20);
                            }
                        }

                        gradFilters[f].setByIndex(d, h, w, sum); // записываем значения градиента
                    }
                }
            }

            // наращиваем градиенты смещения
            for (int h = 0; h < deltasOut.getHeight(); h++) {
                for (int w = 0; w < deltasOut.getWidth(); w++)
                    gradBiases[f] = gradBiases[f].add(deltasOut.getByIndex(f, h, w), mathContext20);
            }
        }
    }

    public void updateWeights(BigDecimal epsilon, BigDecimal alpha) {
        // проходимся по фильтрам
        for (int f = 0; f < this.filtersCount; f++) {
            for (int d = 0; d < this.filters[f].getDepth(); d++) {
                for (int h = 0; h < this.filterSize; h++) {
                    for (int w = 0; w < this.filterSize; w++) {

                        // расчитываем новую разницу между старыми и новыми весами фильтров
                        this.deltaFiltersWeights[f].setByIndex(d, h, w,
                                (epsilon.multiply(this.gradFilters[f].getByIndex(d, h, w), mathContext20))
                                        .add((alpha.multiply(this.deltaFiltersWeights[f].getByIndex(d, h, w), mathContext20)), mathContext20));

                        // обновляем веса
                        this.filters[f].setByIndex(d, h, w, this.filters[f].getByIndex(d, h, w)
                                .add(this.deltaFiltersWeights[f].getByIndex(d, h, w), mathContext20));

                        this.gradFilters[f].setByIndex(d, h, w, BigDecimal.ZERO); // обнуляем градиенты фильтров
                    }
                }
            }

            // расчитываем новую разницу между старыми и новыми весами смещений
            this.deltaBiasesWeights[f] = (epsilon.multiply(this.gradBiases[f], mathContext20))
                    .add((alpha.multiply(this.deltaBiasesWeights[f], mathContext20)), mathContext20);

            // обновляем смещения
            this.biases[f] = this.biases[f].add((epsilon.multiply(this.gradBiases[f], mathContext20))
                    .add(alpha.multiply(this.deltaBiasesWeights[f], mathContext20), mathContext20), mathContext20);

            this.gradBiases[f] = BigDecimal.ZERO; // обнуляем градиенты смещений
        }
    }

    public final TensorSize getOutputSize() {
        return this.outputSize;
    }
}
