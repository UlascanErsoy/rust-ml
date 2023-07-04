use std::{error::Error, io, process};
use std::fs::File;
use matrix_simp::Matrix;

#[derive(Debug, serde::Deserialize)]
struct Candidate {
    years_of_experience:f32,
    functional_competency_score:f32,
    top1_skills_score:f32,
    top2_skills_score:f32,
    top3_skills_score:f32,
    behavior_competency_score:f32,
    top1_behavior_skill_score:f32,
    top2_behavior_skill_score:f32,
    top3_behavior_skill_score:f32,
    call_for_interview:f32

}
fn read_csv(path: String) -> Result<(Matrix<f32>, Matrix<f32>) , Box<dyn Error>> {
    let file = File::open(path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    
    let mut data: Vec<Vec<f32>> = Vec::new();
    let mut res: Vec<Vec<f32>> = Vec::new();

    for result in rdr.deserialize() {
        let candidate: Candidate = result?;
        
        data.push(vec![
                    1_f32,
                    candidate.top3_skills_score,
                    candidate.top2_skills_score,
                    candidate.years_of_experience,
                    candidate.functional_competency_score,
                    candidate.top1_skills_score,
                    candidate.behavior_competency_score,
                    candidate.top1_behavior_skill_score,
                    candidate.top2_behavior_skill_score,
                    candidate.top3_behavior_skill_score
        ]);

        res.push(vec![candidate.call_for_interview]);
    }
    
    Ok((Matrix::from(data), Matrix::from(res)))
}

fn init_hypothesis(features: usize) -> Matrix<f32> {
   Matrix::new(1_f32, features, 1) 
}

fn apply_learning<'a>(alpha: f32,
                      inputs: Matrix<f32>,
                      theta: &'a mut Matrix<f32>,
                      result: &Matrix<f32>) -> &'a mut Matrix<f32> {
   
    let res = result.clone();
    let mut cost: Vec<f32> = vec![0_f32; inputs.m];


    for idx in 0..inputs.n {
        let data: Vec<Vec<f32>> = vec![Vec::from(inputs.get_row(idx))];
        let x: Matrix<f32> = Matrix::from(data).transpose();

        let mut h = (((theta.transpose() * x.clone()) * -1_f32).exp() + 1_f32).one_over();

       
        for fdx in 0..theta.n {
                cost[fdx] = cost[fdx] + (res[idx] - h[0]) * x[fdx];
        }
       //println!("Error: {} - {} = {}", res[idx], h[0],res[idx] - h[0]);
    }
    
    for fdx in 0..theta.n {
        theta[fdx] = theta[fdx] + (alpha / (inputs.n as f32)) * cost[fdx];
    }
    //println!("{:?}", theta);
    theta
}

fn predict(inputs: Matrix<f32>, theta: Matrix<f32>) -> Matrix<f32> {
    let mut res: Vec<f32> = Vec::new();

    for idx in 0..inputs.n {
        let data: Vec<Vec<f32>> = vec![Vec::from(inputs.get_row(idx))];
        let x: Matrix<f32> = Matrix::from(data).transpose();
        let mut h = (((theta.transpose() * x.clone()) * -1_f32).exp() + 1_f32).one_over();
                
        if h[0] >= 0.5 {
            h[0] = 1_f32;
        }else{
            h[0] = 0_f32;
        }
        
        res.push(h[0]);    
    }

    Matrix::from(vec![res])
}

fn main() {
    
    let (inputs, output) = read_csv(String::from("../data/hr/train.csv")).unwrap();

    let mut theta = init_hypothesis(inputs.m);

    for _ in 0..1000 {
        apply_learning(0.01_f32, inputs.clone(), &mut theta, &output);
    }
    
    
    println!("Theta: {:?}", theta);

    let (v_in, v_out) = read_csv(String::from("../data/hr/test.csv")).unwrap();
    let prediction = predict(v_in, theta);
    let correct = prediction.get_row(0)
                            .iter()
                            .zip(v_out.get_col(0).iter())
                            .fold(0_u32,|mut acc,(a,b)| {
                                if a == b {
                                    acc +=1;
                                }

                                acc
                            });
    
    let tots = prediction.m as u32;
    println!("{correct}/{tots}: {}%",100 * correct / tots);

}
