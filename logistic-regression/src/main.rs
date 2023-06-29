use std::{error::Error, io, process};
use std::fs::File;
use matrix_simp::Matrix;

#[derive(Debug, serde::Deserialize)]
struct Passenger {
    PassengerId: String,
    HomePlanet: String,
    CryoSleep: String,
    Cabin: String,
    Destination: String,
    Age: Option<f32>,
    VIP: String,
    RoomService: Option<f32>,
    FoodCourt: Option<f32>,
    ShoppingMall: String,
    Spa: Option<f32>,
    VRDeck: Option<f32>,
    Name: String
}

fn main() {

    let file = File::open("data/train.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);

    for result in rdr.deserialize() {
        let passenger: Passenger = result.unwrap();
    }


    
    let data : &[&[f32]] = &[&[0_f32,1_f32], &[1_f32, 0_f32], &[1_f32, 0_f32]];
    let mat: Matrix<f32> = Matrix::from(data);
    println!("{:?}", mat);
    println!("\n+\n");
    let data2 : &[&[f32]] = &[&[1_f32,1_f32], &[1_f32, 1_f32], &[0_f32, 1_f32]];
    let mat2: Matrix<f32> = Matrix::from(data2);
    println!("{:?}", mat2);
    println!("\n=\n");
    println!("{:?}", mat * mat2);

}
