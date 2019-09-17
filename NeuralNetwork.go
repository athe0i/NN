package NN

import (
  "gonum.org/v1/gonum/mat"
  "math"
  "gonum.org/v1/gonum/stat/distuv"
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
	// forward propagation - we need intermediate results for learning
	inputs := mat.NewDense(len(inputData), 1, inputData)
  hiddenInputs := make([]mat.Matrix, len(net.hiddens))
	hiddenInputs[0] = dot(net.hiddenWeights[0], inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs[0])

  outputs := make([]mat.Matrix, len(net.hiddens))
  //copy(hiddenOutputs, outputs[0])
  outputs[0] = apply(sigmoid, hiddenInputs[0])

  for i := 1; i < len(net.hiddenWeights); i++ {
    hiddenInputs[i] = dot(net.hiddenWeights[i], hiddenOutputs)
    hiddenOutputs = apply(sigmoid, hiddenInputs[i])
    //copy(hiddenOutputs, outputs[i])
    outputs[i] = apply(sigmoid, hiddenInputs[i])
  }

	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
  hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	// backpropagate
  //fmt.Println("Backprop output")
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenInputs[len(outputs) - 1].T()))).(*mat.Dense)

  if (len(net.hiddens) > 1) {
    //fmt.Println("Backprop hidden except first")
    for i := len(net.hiddenWeights) - 1; i > 0; i-- {
      nextErrors := dot(net.hiddenWeights[i].T(), hiddenErrors)
      net.hiddenWeights[i] = add(net.hiddenWeights[i],
        scale(net.learningRate,
          dot(multiply(hiddenErrors, sigmoidPrime(outputs[i])),
            hiddenInputs[i-1].T()))).(*mat.Dense)

      hiddenErrors = nextErrors
    }
  }

  net.hiddenWeights[0] = add(net.hiddenWeights[0],
    scale(net.learningRate,
      dot(multiply(hiddenErrors, sigmoidPrime(outputs[0])),
        inputs.T()))).(*mat.Dense)
}

func (net NeuralNetwork) Save() {
	/*h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}*/
}

// load a neural network from file
func (net NeuralNetwork) Load() {
  /*fmt.Println("Loading")
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}

  fmt.Println(net.hiddenWeights)
	return*/
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

func subtract(m, n mat.Matrix) mat.Matrix {
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
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}
