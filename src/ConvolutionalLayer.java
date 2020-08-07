import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Vector;

public class ConvolutionalLayer {

    private final MathContext mathContext20 = new MathContext(20);

    private TensorSize inputSize = new TensorSize(); // размер входа
//    private TensorSize inputSize; // размер входа
    private TensorSize outputSize = new TensorSize(); // размер выхода
//    private TensorSize outputSize; // размер выхода

    private Vector<Tensor> filters; // фильтры
    private Vector<BigDecimal> biases; // смешения

    private Vector<Tensor> gradFilters; // градиенты фильтров
    private Vector<BigDecimal> gradBiases; // градиенты смещений

    private int padding; // дополнение нулями
    private int stride; // шаг свертки

    private int filtersCount; // кол-во фильтров
    private int filterSize; // размер фильтров
    private int filterDepth; // глубина фильтров

    // инициализация весовых коэффициентов
    private void initWeights() {
        // проходимся по каждому из фильтров
        for (int i = 0; i < filtersCount; i++) {
            for (int h = 0; h < filterSize; h++) {
                for (int w = 0; w < filterSize; w++) {
                    for (int d = 0; d < filterDepth; d++)
                        filters.get(i).setByIndex(d, h, w, BigDecimal.valueOf(Math.random()));
                }
            }
            biases.set(i, BigDecimal.valueOf(0.01)); // смещения устанавливаем в 0.01
        }
    }

    public ConvolutionalLayer(TensorSize tensorSize,
                              int filtersCount,
                              int filterSize,
                              int padding,
                              int stride) {
        // запоминаем входной размер
        inputSize.width = tensorSize.width;
        inputSize.height = tensorSize.height;
        inputSize.depth = tensorSize.depth;

        // вычисляем выходной размер
        outputSize.width = (tensorSize.width - filterSize + 2*padding)/stride + 1;
        outputSize.height = (tensorSize.height - filterSize + 2*padding)/stride + 1;
        outputSize.depth = filtersCount;

        this.padding = padding; // сохраняем дополнение нулями
        this.stride = stride; // сохраняем шаг свертки

        this.filtersCount = filtersCount; // сохраняем кол-во фильтров
        this.filterSize = filterSize; // сохраняем размер фильтров
        this.filterDepth = tensorSize.depth; // сохраняем глубину фильтров

        // добавсяем filtersCount тензоров для весов фильтров и их градиентов
        filters = new Vector<>(filtersCount);
        gradFilters = new Vector<>(filtersCount);
        for (int i = 0; i < filtersCount; i++) {
            filters.add(new Tensor(filterSize, filterSize, filterDepth));
//            filters.set(i, new Tensor(filterSize, filterSize, filterDepth));
            gradFilters.add(new Tensor(filterSize, filterSize, filterDepth));
//            gradFilters.set(i, new Tensor(filterSize, filterSize, filterDepth));
        }

        // добавляем filtersCount нулей для весов фильтров и их градиентов
        biases = new Vector<>(filtersCount);
        gradBiases = new Vector<>(filtersCount);
        for (int i = 0; i < filtersCount; i++) {
            biases.add(BigDecimal.ZERO);
//            biases.set(i, BigDecimal.ZERO);
            gradBiases.add(BigDecimal.ZERO);
//            gradBiases.set(i, BigDecimal.ZERO);
        }

        initWeights();
    }

    // прямое распространение
    public Tensor forward(final Tensor input) {
        Tensor output = new Tensor(outputSize); // создаем выходной тензор

        // проходимся по каждому из фильтров
        for (int f = 0; f < filtersCount; f++) {
            for (int h = 0; h < outputSize.height; h++) {
                for (int w = 0; w < outputSize.width; w++) {
                    BigDecimal sum = biases.get(f); // сразу прибавляем смещение

                    // проходимся по фильтрам
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            int i0 = stride*h + i - padding;
                            int j0 = stride*w + j - padding;

                            // поскольку вне границ входного тензора элементы нулевые, то игнорируем их
                            if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                                continue;

                            // проходимся по всей глубине тензора и считаем сумму
                            for (int d = 0; d < filterDepth; d++)
                                sum = sum.add(input.getByIndex(d, i0, j0).multiply(filters.get(f).getByIndex(d, i,j), mathContext20), mathContext20);
                        }
                    }

                    output.setByIndex(f, h, w, sum); // записываем результат свертки в выходной тензор
                }
            }
        }

        return output; // возвращаем выходной тензор
    }

    // обратное распространение
    public Tensor backward(final Tensor dOut, final Tensor input) {
        TensorSize deltaSize = new TensorSize(); // размер дельт

        // расчитываем размер для дельт
        deltaSize.height = stride*(outputSize.height - 1) + 1;
        deltaSize.width = stride*(outputSize.width - 1) + 1;
        deltaSize.depth = outputSize.depth;

        Tensor deltas = new Tensor(deltaSize); // создаем тензор для дельт

        // расчитываем значения для дельт
        for (int d = 0; d < deltaSize.depth; d++) {
            for (int h = 0; h < outputSize.height; h++) {
                for (int w = 0; w < outputSize.width; w++)
                    deltas.setByIndex(d, h*stride, w*stride, dOut.getByIndex(d, h, w));
            }
        }

        // расчитываем градиенты весов фильтров и смещений
        for (int f = 0; f < filtersCount; f++) {
            for (int h = 0; h < deltaSize.height; h++) {
                for (int w = 0; w < deltaSize.width; w++) {
                    BigDecimal delta = deltas.getByIndex(f, h, w); // запоминаем значение градиента

                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            int i0 = i + h - padding;
                            int j0 = j + w - padding;

                            // игнорируем выходящие за границы элементы
                            if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                                continue;

                            // наращиваем градиент фильтра
                            for (int d = 0; d < filterDepth; d++)
                                gradFilters.get(f).setByIndex(d, i, j, gradFilters.get(f).getByIndex(d, i, j).add(delta.multiply(input.getByIndex(d, i0, j0), mathContext20), mathContext20));

                            gradBiases.set(f, gradBiases.get(f).add(delta, mathContext20)); // наращиваем градиент смещения
                        }
                    }
                }
            }
        }

        int pad = filterSize - 1 - padding; // величина дополнения
        Tensor gradInput = new Tensor(inputSize); // создаем тензор градиентов по входу

        // расчитываем значения градиента
        for (int h = 0; h < inputSize.height; h++) {
            for (int w = 0; w < inputSize.width; w++) {
                for (int d = 0; d < filterDepth; d++) {
                    BigDecimal sum = BigDecimal.ZERO; // сумма для градиента

                    // идем по всем весовім коєффициентам фильтров
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            int i0 = h + i - pad;
                            int j0 = w + j - pad;

                            // игнорируем выходящие за границы элементы
                            if (i0 < 0 || i0 >= deltaSize.height || j0 < 0 || j0 >= deltaSize.width)
                                continue;

                            // суммируем по всем фильтрам
                            for (int f = 0; f < filtersCount; f++) // добавляем произведение повернутых фильтров на дельты
                                sum = sum.add(filters.get(f).getByIndex(d, filterSize - 1 - i, filterSize - 1 - j).multiply(deltas.getByIndex(f, i0, j0), mathContext20), mathContext20);
                        }
                    }

                    gradInput.setByIndex(d, h, w, sum); // записываем результат в тензор градиента
                }
            }
        }

        return gradInput; // возвращаем тензор градиента
    }

    /*
     * Данный алгоритм является наиболее простым и довольно медленным с точки зрения сходимости методом оптимизации,
     * однако ничего не мешает реализовать более быстрые алгоритмы, как, например, Adam, Adagrad, RMSprop
     */
    // обновление весовых коэффициентов
    public void updateWeights(BigDecimal learningRate/*, String compareTo*/) {
        for (int f = 0; f < filtersCount; f++) {
            for (int h = 0; h < filterSize; h++) {
                for (int w = 0; w < filterSize; w++) {
                    for (int d = 0; d < filterDepth; d++) {

                        /*// вычитаем градиент, умноженный на скорость обучения
                        if (compareTo.equals(">"))
                            filters.get(f).setByIndex(d, h, w, filters.get(f).getByIndex(d, h, w).subtract(learningRate.multiply(gradFilters.get(f).getByIndex(d, h, w), mathContext20), mathContext20));
                        else
                            filters.get(f).setByIndex(d, h, w, filters.get(f).getByIndex(d, h, w).add(learningRate.multiply(gradFilters.get(f).getByIndex(d, h, w), mathContext20), mathContext20));*/

                        // вычитаем градиент, умноженный на скорость обучения
                        filters.get(f).setByIndex(d, h, w, filters.get(f).getByIndex(d, h, w).subtract(learningRate.multiply(gradFilters.get(f).getByIndex(d, h, w), mathContext20), mathContext20));
                        // обнуляем градиент фильтра
                        gradFilters.get(f).setByIndex(d, h, w, BigDecimal.ZERO);
                    }
                }
            }

            /*if (compareTo.equals(">"))
                biases.set(f, biases.get(f).subtract(learningRate.multiply(gradBiases.get(f), mathContext20), mathContext20)); // вычитаем градиент, умноженный на скорость обучения
            else
                biases.set(f, biases.get(f).add(learningRate.multiply(gradBiases.get(f), mathContext20), mathContext20)); // вычитаем градиент, умноженный на скорость обучения*/

            biases.set(f, biases.get(f).subtract(learningRate.multiply(gradBiases.get(f), mathContext20), mathContext20)); // вычитаем градиент, умноженный на скорость обучения
            gradBiases.set(f, BigDecimal.ZERO); // обнуляем градиент веса смещения
        }
    }

/*    // установка веса фильтра по индексу
    public void setWeight(int f, int d, int h, int w, BigDecimal weight) {
        filters.get(f).setByIndex(d, h, w, weight);
    }

    // установка веса смещения по индексу
    public void setBias(int f, BigDecimal bias) {
        biases.set(f, bias);
    }

    public static void main(String[] args) {

        TensorSize inputSize = new TensorSize(3, 5, 5);
        ConvolutionalLayer convLayer = new ConvolutionalLayer(inputSize, 2, 3, 1, 2);

        convLayer.setWeight(0, 0, 0, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 0, 1, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 1, 0, 1, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 1, 1, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 1, 1, 2, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 2, 0, 1, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 2, 1, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 2, 1, 2, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 2, 2, 2, BigDecimal.valueOf(-1));
        convLayer.setWeight(0, 0, 0, 1, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 0, 0, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 0, 1, 1, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 0, 1, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 0, 2, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 1, 0, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 1, 2, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 2, 0, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(0, 2, 2, 1, BigDecimal.valueOf(1));

        convLayer.setWeight(1, 0, 0, 2, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 0, 2, 1, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 1, 0, 1, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 1, 1, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 1, 2, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 2, 1, 1, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 2, 2, 0, BigDecimal.valueOf(-1));
        convLayer.setWeight(1, 0, 1, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 1, 0, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 1, 1, 1, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 1, 2, 1, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 1, 2, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 2, 0, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 2, 0, 2, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 2, 1, 0, BigDecimal.valueOf(1));
        convLayer.setWeight(1, 2, 1, 2, BigDecimal.valueOf(1));

        for (int f = 0; f < convLayer.filtersCount; f++) {
            for (int d = 0; d < convLayer.filterDepth; d++) {
                for (int h = 0; h < convLayer.filterSize; h++) {
                    for (int w = 0; w < convLayer.filterSize; w++) {
                        if (!convLayer.filters.get(f).getByIndex(d, h, w).equals(BigDecimal.ONE)
                                && !convLayer.filters.get(f).getByIndex(d, h, w).equals(BigDecimal.valueOf(-1)))
                            convLayer.setWeight(f, d, h, w, BigDecimal.ZERO);
                    }
                }
            }
        }

//        convLayer.filters.get(0).print();
//        System.out.println();
//        System.out.println();
//        System.out.println();
//        convLayer.filters.get(1).print();
//        System.exit(0);

        convLayer.setBias(0, BigDecimal.ONE);
        convLayer.setBias(1, BigDecimal.ZERO);

        Tensor input = new Tensor(5, 5, 3);
        input.setByIndex(0, 0, 0, BigDecimal.ONE);
        input.setByIndex(0, 0, 3, BigDecimal.ONE);
        input.setByIndex(0, 1, 4, BigDecimal.ONE);
        input.setByIndex(0, 2, 0, BigDecimal.ONE);
        input.setByIndex(0, 3, 4, BigDecimal.ONE);
        input.setByIndex(0, 4, 2, BigDecimal.ONE);
        input.setByIndex(0, 4, 4, BigDecimal.ONE);
        input.setByIndex(1, 0, 0, BigDecimal.ONE);
        input.setByIndex(1, 0, 3, BigDecimal.ONE);
        input.setByIndex(1, 2, 0, BigDecimal.ONE);
        input.setByIndex(1, 2, 3, BigDecimal.ONE);
        input.setByIndex(1, 2, 4, BigDecimal.ONE);
        input.setByIndex(1, 3, 3, BigDecimal.ONE);
        input.setByIndex(1, 4, 2, BigDecimal.ONE);
        input.setByIndex(2, 0, 3, BigDecimal.ONE);
        input.setByIndex(2, 0, 4, BigDecimal.ONE);
        input.setByIndex(2, 1, 3, BigDecimal.ONE);
        input.setByIndex(2, 1, 4, BigDecimal.ONE);
        input.setByIndex(2, 2, 2, BigDecimal.ONE);
        input.setByIndex(2, 2, 4, BigDecimal.ONE);
        input.setByIndex(2, 3, 1, BigDecimal.ONE);
        input.setByIndex(2, 3, 3, BigDecimal.ONE);
        input.setByIndex(2, 4, 0, BigDecimal.ONE);
        input.setByIndex(2, 4, 2, BigDecimal.ONE);
        input.setByIndex(2, 4, 4, BigDecimal.ONE);
        input.setByIndex(0, 0, 1, BigDecimal.valueOf(2));
        input.setByIndex(0, 1, 0, BigDecimal.valueOf(2));
        input.setByIndex(0, 2, 1, BigDecimal.valueOf(2));
        input.setByIndex(0, 2, 2, BigDecimal.valueOf(2));
        input.setByIndex(0, 2, 4, BigDecimal.valueOf(2));
        input.setByIndex(0, 3, 0, BigDecimal.valueOf(2));
        input.setByIndex(0, 3, 1, BigDecimal.valueOf(2));
        input.setByIndex(0, 3, 2, BigDecimal.valueOf(2));
        input.setByIndex(0, 4, 0, BigDecimal.valueOf(2));
        input.setByIndex(1, 0, 1, BigDecimal.valueOf(2));
        input.setByIndex(1, 0, 2, BigDecimal.valueOf(2));
        input.setByIndex(1, 0, 4, BigDecimal.valueOf(2));
        input.setByIndex(1, 1, 1, BigDecimal.valueOf(2));
        input.setByIndex(1, 1, 2, BigDecimal.valueOf(2));
        input.setByIndex(1, 1, 4, BigDecimal.valueOf(2));
        input.setByIndex(1, 2, 1, BigDecimal.valueOf(2));
        input.setByIndex(1, 2, 2, BigDecimal.valueOf(2));
        input.setByIndex(1, 3, 0, BigDecimal.valueOf(2));
        input.setByIndex(1, 3, 1, BigDecimal.valueOf(2));
        input.setByIndex(1, 4, 0, BigDecimal.valueOf(2));
        input.setByIndex(1, 4, 1, BigDecimal.valueOf(2));
        input.setByIndex(2, 0, 1, BigDecimal.valueOf(2));
        input.setByIndex(2, 1, 0, BigDecimal.valueOf(2));
        input.setByIndex(2, 1, 2, BigDecimal.valueOf(2));
        input.setByIndex(2, 2, 0, BigDecimal.valueOf(2));
        input.setByIndex(2, 2, 3, BigDecimal.valueOf(2));
        input.setByIndex(2, 3, 4, BigDecimal.valueOf(2));
        input.setByIndex(2, 4, 3, BigDecimal.valueOf(2));

        convLayer.forward(input).print();
    }*/
}
