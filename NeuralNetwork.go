package NN

import (
  "gonum.org/v1/gonum/mat"
  "math"
  "gonum.org/v1/gonum/stat/distuv"
  "fmt"
  "os"
  )

type NeuralNetwork struct {
  inputs        int
	hiddens       []int
	outputs       int
	hiddenWeights []*mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

func NewNeuralNetwork(input int, hidden []int, output int, rate float64) (net NeuralNetwork) {
	net = NeuralNetwork{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}

  net.populateWeights()

	return
}

func (net *NeuralNetwork) populateWeights() {
  net.hiddenWeights = make([]*mat.Dense, len(net.hiddens))
  if (len(net.hiddens) < 2) {
    net.outputWeights = mat.NewDense(net.outputs, net.hiddens[0], randomArray(net.outputs*net.hiddens[0], float64(net.hiddens[0])))
  } else {
    net.outputWeights = mat.NewDense(net.outputs, net.hiddens[len(net.hiddens) - 1], randomArray(net.outputs*net.hiddens[len(net.hiddens) - 1], float64(net.inputs)))
  }

  for i, curHiddens := range net.hiddens {
    if (i == 0) {
      net.hiddenWeights[i] = mat.NewDense(curHiddens, net.inputs, randomArray(curHiddens*net.inputs, float64(net.inputs)))
    } else {
      net.hiddenWeights[i] = mat.NewDense(curHiddens, net.hiddens[i - 1], randomArray(curHiddens*net.hiddens[i-1], float64(net.hiddens[i-1])))
    }
  }
}

func (net NeuralNetwork) GetInputsNumber() int {
  return net.inputs
}

func (net NeuralNetwork) GetOutputsNumber() int {
  return net.outputs
}

func (net NeuralNetwork) Predict(inputData []float64) mat.Matrix {
	// forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights[0], inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)

  for i := 1; i < len(net.hiddenWeights); i++ {
    hiddenInputs = dot(net.hiddenWeights[i], hiddenOutputs)
    hiddenOutputs = apply(sigmoid, hiddenInputs)
  }

	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	return finalOutputs
}

func (net *NeuralNetwork) Train(inputData []float64, targetData []float64) {
  inputs := mat.NewDense(len(inputData), 1, inputData)
  hiddenInputs := make([]mat.Matrix, len(net.hiddens))
  hiddenOutputs := make([]mat.Matrix, len(net.hiddens))

  hiddenInputs[0] = dot(net.hiddenWeights[0], inputs)
  hiddenOutputs[0] = apply(sigmoid, hiddenInputs[0])

  for i := 1; i < len(net.hiddens); i++ {
    hiddenInputs[i] = dot(net.hiddenWeights[i], hiddenOutputs[i-1])
    hiddenOutputs[i] = apply(sigmoid, hiddenInputs[i])
  }

  outputInputs := dot(net.outputWeights, hiddenOutputs[len(hiddenOutputs) - 1])
  outputOutputs := apply(sigmoid, outputInputs)

  //backpropagate
  targets := mat.NewDense(len(targetData), 1, targetData)
  outputErrors := substract(targets, outputOutputs)
  hiddenErrors := dot(net.outputWeights.T(), outputErrors)

  net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(outputOutputs)),
				hiddenOutputs[len(hiddenOutputs) - 1].T()))).(*mat.Dense)

  for i := len(net.hiddenWeights) - 1; i > 0; i-- {
    net.hiddenWeights[i] = add(net.hiddenWeights[i],
		  scale(net.learningRate,
			  dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs[i])),
				  hiddenInputs[i].T()))).(*mat.Dense)

    hiddenErrors = dot(net.hiddenWeights[i].T(), hiddenErrors)
  }
  net.hiddenWeights[0] = add(net.hiddenWeights[0],
    scale(net.learningRate,
      dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs[0])),
        inputs.T()))).(*mat.Dense)
}

func (net NeuralNetwork) Save() {
  path := fmt.Sprintf("data/%s", net.getDataDirName())
  fmt.Println(path)
  if _, err := os.Stat(path); os.IsNotExist(err) {
    os.Mkdir(path, 0755)
  }

  for i,layerWeights := range net.hiddenWeights {
    h, err := os.Create(fmt.Sprintf("%s/hweights%d.model", path, i))
  	defer h.Close()
  	if err == nil {
  		layerWeights.MarshalBinaryTo(h)
  	}
  }

	o, err := os.Create(fmt.Sprintf("%s/oweights.model", path))
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

// load a neural network from file
func (net NeuralNetwork) Load() {
  fmt.Println("Loading")
  path := fmt.Sprintf("data/%s", net.getDataDirName())

  for i,_ := range net.hiddens {
    h, err := os.Open(fmt.Sprintf("%s/hweights%d.model", path, i))
  	defer h.Close()
  	if err == nil {
  		net.hiddenWeights[i].Reset()
  		net.hiddenWeights[i].UnmarshalBinaryFrom(h)
  	}
  }

	o, err := os.Open(fmt.Sprintf("%s/oweights.model", path))
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}

	return
}

func (net NeuralNetwork) getDataDirName() string {
  dirName := fmt.Sprintf("%d-", net.inputs)

  for _,layerNeurons := range net.hiddens {
    dirName += fmt.Sprintf("%d-", layerNeurons)
  }

  return fmt.Sprintf("%s%d", dirName, net.outputs)
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
  /*fmt.Println("dot dims")
  fmt.Println(m.Dims())
  fmt.Println(n.Dims())*/
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
  /*fmt.Println("add dims")
  fmt.Println(m.Dims())
  fmt.Println(n.Dims())*/
	o.Add(m, n)
	return o
}

func substract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, substract(ones, m)) // m * (1 - m)
}
