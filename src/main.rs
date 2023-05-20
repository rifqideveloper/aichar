use std::thread;
use std::sync::mpsc;

use std::sync::mpsc::Sender;
static mut MESSAGE_BUFFER:[i16;511] = [0;511];
static mut TOKEN : Vec<&i16> = Vec::new();
static mut INPUT : Vec<Vec<&i16>> = Vec::new();
static mut MATRIX_2_D : Vec<Vec<i16>> = Vec::new();
static mut NEURON_CHANNEL :Vec<(Sender<Option<f32>>, std::sync::mpsc::Receiver<Option<f32>>)>= Vec::new();
fn message_buffer_gen() {
	for i in 0..511 {
		unsafe {
			MESSAGE_BUFFER[i] = i as i16 - 255
		}
	}
}
fn neuron_input(msg:usize,send:Vec<usize>,weight:f32,biase:f32) -> Option<f32>{
	
	while true {
		if let Some(msg) = unsafe{NEURON_CHANNEL[msg].1.recv().unwrap()} {
			for i in &send {
				unsafe {
					NEURON_CHANNEL[*i].0.send(Some(
						msg  * weight + biase
					)).unwrap()
				}
			}
		} else {
			for i in send {
				unsafe {
					NEURON_CHANNEL[i].0.send(None).unwrap()
				}
			}
			return None
		}
	}
	
	return None
}
fn neuron_hidden(msg:usize,send:Vec<usize>,weight:f32) -> Option<f32>{
	while true {
		if let Some(msg) = unsafe{NEURON_CHANNEL[msg].1.recv().unwrap()} {
			for i in &send {
				unsafe {
					NEURON_CHANNEL[*i].0.send(Some(
						msg  * weight 
					)).unwrap()
				}
			}
		} else {
			for i in send {
				unsafe {
					NEURON_CHANNEL[i].0.send(None).unwrap()
				}
			}
			return None
		}
	}
	return None
}
fn neuron_output(msg:usize,biase:f32) -> Option<f32>{
	let mut v = Vec::new();
	while true {
		if let Some(msg) = unsafe{NEURON_CHANNEL[msg].1.recv().unwrap()} {
			v.push(msg);
		} else {
			return Some(v.iter().sum::<f32>() - biase );
		}
	}
	return None
}
fn matrix(input:Vec<f32>,hidden:usize,out:usize) -> (usize,usize,Vec<f32>) {
	let mut x = input.len() + hidden + out ;
	let mut tr = Vec::with_capacity(x);
	unsafe {
		NEURON_CHANNEL.clear();
		for i in 0..x {
			NEURON_CHANNEL.push(mpsc::channel::<Option<f32>>());
			unsafe{
				if i < input.len() {
					NEURON_CHANNEL[i].0.send(Some( input[i] as f32 )).unwrap();
				}
			}
			NEURON_CHANNEL[i].0.send(None).unwrap();
		}
	}
	
	x = input.len() + hidden;
	for i in 0..input.len() {
		let msg : usize = i;
		let mut send : Vec<usize>= Vec::new();
		for ii in input.len()..x {
			send.push(ii)
		}
		let thread_join_handle = thread::spawn(move || {
			
			neuron_input(msg,send,-34.4,1.14)
			
		});
		tr.push(thread_join_handle);
		
	}
	
	let cc = x;
	x += out;
	for i in cc..x {
		let msg : usize= i;
		let mut send : Vec<usize>= Vec::new();
		for i in 0..out {
			unsafe { send.push(NEURON_CHANNEL.len()-(i + 1)) }
		}
		let thread_join_handle = thread::spawn(move || {
			neuron_hidden(msg,send,0f32)
		});
		tr.push(thread_join_handle)
	}
	
	for i in 0..out {
		let msg : usize= unsafe { NEURON_CHANNEL.len()-(i + 1) };
		let thread_join_handle = thread::spawn(move || {
			neuron_output(msg,0f32)
		});
		tr.push(thread_join_handle)
	}
	let mut output : Vec<f32> = Vec::new();
	for thread_join_handle in tr {
		if let Some(output_n) = thread_join_handle.join().unwrap() {
			output.push(output_n)
		}
	}
	println!("out:{:#?}",output);
	return (hidden,out,output)
}
fn tokenize(strs:String) {
	
	for strs in strs.split(" ") {
		for i in strs.as_bytes() {
			unsafe{TOKEN.push(
				&MESSAGE_BUFFER[(*i as usize) + 255]
			)}
		}
		unsafe {
			INPUT.push(TOKEN.clone());
			TOKEN.clear();
		}
	}
	
}
fn main() {
	message_buffer_gen();
    //println!("BUFFER{:?}",unsafe{ MESSAGE_BUFFER});
	tokenize("hallo".to_string());
	//println!("TOKEN{:?}",unsafe{&TOKEN});
	//println!("INPUT{:?}",unsafe{&INPUT});
	matrix(Vec::from([12f32,24f32]),2,3);
}
