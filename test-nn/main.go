package main

import (
  "fmt"
  "mnist"
  "time"
  "flag"
  "NN"
  "math/rand"
)

func main() {
  // 784 inputs - 28 x 28 pixels, each pixel is an input
  // 200 hidden neurons - an arbitrary number
  // 10 outputs - digits 0 to 9
  // 0.1 is the learning rate
  hiddens := []int{200}
  net := NN.NewNeuralNetwork(784, hiddens, 10, 0.1)

  mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
  flag.Parse()

  // train or mass predict to determine the effectiveness of the trained network
  switch *mnist {
  case "train":
    train(&net)
    net.Save()
  case "predict":
    net.Load()
    predict(&net)
  case "tp":
    train(&net)
    net.Save()
    predict(&net)
  default:
    // don't do anything
  }
}

func predict(net *NN.NeuralNetwork) {
  fmt.Println("Predicting")
  start := time.Now()
  dataSet, err := mnist.ReadTestSet("./mnist")
  score := 0

  if err != nil {
    fmt.Println(err)
    return
  }

  for i := 0; i < dataSet.N; i++ {
    image := dataSet.Data[i].Image

    inputs := getInputsFromImage(image, net.GetInputsNumber())
    outputs := net.Predict(inputs)
    best := 0
    highest := 0.0

    for i := 0; i < net.GetOutputsNumber(); i++ {
      if outputs.At(i, 0) > highest {
        best = i
        highest = outputs.At(i, 0)
      }
    }

    label := dataSet.Data[i].Digit
    if best == label {
      score++
    } else {
      fmt.Println("Predict", best, " Expected", label, "OUT", outputs)
      mnist.PrintImage(image)
    }
  }

  elapsed := time.Since(start)
  fmt.Printf("Time taken to check: %s\n", elapsed)
  fmt.Println("score:", score)
}

func train(net *NN.NeuralNetwork) {
  fmt.Println("Train")
  dataSet, err := mnist.ReadTrainSet("./mnist")


  if err != nil {
    fmt.Println(err)
    return
  }
  rand.Seed(time.Now().UTC().UnixNano())

  for epochs := 0; epochs < 2; epochs++ {
    fmt.Println("Epoch ", epochs)
    start := time.Now()


    for i := 0; i < dataSet.N; i++ {
      //fmt.Println("Sample ", i)
      image := dataSet.Data[i].Image

      inputs := getInputsFromImage(image, net.GetInputsNumber())

      //fmt.Println(inputs)

      targets := make([]float64, 10)
      for i := range targets {
        targets[i] = 0.01
      }
      x := dataSet.Data[i].Digit
      targets[x] = 0.99

      net.Train(inputs, targets)
    }


      elapsed := time.Since(start)
      fmt.Printf("\nTime taken to train: %s\n", elapsed)

    }
}

func getInputsFromImage(image [][]uint8, inputNum int) []float64 {
  inputs := make([]float64, inputNum)
  counter := -1

  for _, row := range image {
    //fmt.Println(row)

    for _, pix := range row {
      counter++

      //x, _ := strconv.ParseFloat(string(pix), 64)
      x := float64(pix)

      if x == 0 {
        x = 1.0
      }

      inputs[counter] = (x / 255.0) + 0.01

    }
  }

  return inputs
}

func printData(dataSet *mnist.DataSet, index int) {
  data := dataSet.Data[index]
  fmt.Println(data.Digit)			// print Digit (label)
  mnist.PrintImage(data.Image)	// print Image
}
