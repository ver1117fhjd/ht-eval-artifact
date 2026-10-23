// Functional models for Trust-Hub Nangate-original cells.
// Combinational models are derived from the nangate45 Liberty `function`
// strings; sequential / compound-only cells are written explicitly.
`timescale 1ns/1ps

module AND2X1(IN1, IN2, Q);
  input IN1; input IN2; output Q;
  assign Q = (IN1&IN2);
endmodule

module AND2X2(IN1, IN2, Q);
  input IN1; input IN2; output Q;
  assign Q = (IN1&IN2);
endmodule

module AND3X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = ((IN1&IN2)&IN3);
endmodule

module AND3X4(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = ((IN1&IN2)&IN3);
endmodule

module AND4X1(IN1, IN2, IN3, IN4, Q);
  input IN1; input IN2; input IN3; input IN4; output Q;
  assign Q = (((IN1&IN2)&IN3)&IN4);
endmodule

module AO21X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = (IN1 & IN2) | IN3;
endmodule

module AO221X1(IN1, IN2, IN3, IN4, IN5, Q);
  input IN1; input IN2; input IN3; input IN4; input IN5; output Q;
  assign Q = (IN1 & IN2) | (IN3 & IN4) | IN5;
endmodule

module AO222X1(IN1, IN2, IN3, IN4, IN5, IN6, Q);
  input IN1; input IN2; input IN3; input IN4; input IN5; input IN6; output Q;
  assign Q = (IN1 & IN2) | (IN3 & IN4) | (IN5 & IN6);
endmodule

module AO22X1(IN1, IN2, IN3, IN4, Q);
  input IN1; input IN2; input IN3; input IN4; output Q;
  assign Q = (IN1 & IN2) | (IN3 & IN4);
endmodule

module AOI21X1(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~(IN1|(IN2&IN3));
endmodule

module AOI221X1(IN1, IN2, IN3, IN4, IN5, QN);
  input IN1; input IN2; input IN3; input IN4; input IN5; output QN;
  assign QN = ~(((IN4&IN5)|IN1)|(IN2&IN3));
endmodule

module AOI222X1(IN1, IN2, IN3, IN4, IN5, IN6, QN);
  input IN1; input IN2; input IN3; input IN4; input IN5; input IN6; output QN;
  assign QN = ~(((IN1&IN2)|(IN3&IN4))|(IN5&IN6));
endmodule

module AOI22X1(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~((IN1&IN2)|(IN3&IN4));
endmodule

module DFFARX1(CLK, D, RSTB, Q, QN);
  input CLK; input D; input RSTB; output reg Q; output QN;
  always @(posedge CLK or negedge RSTB) if (!RSTB) Q <= 1'b0; else Q <= D; assign QN = ~Q;
endmodule

module DFFNX2(CLK, D, Q);
  input CLK; input D; output reg Q;
  always @(negedge CLK) Q <= D;
endmodule

module DFFX2(CLK, D, Q, QN);
  input CLK; input D; output reg Q; output QN;
  always @(posedge CLK) Q <= D; assign QN = ~Q;
endmodule

module FADDX1(A, B, CI, CO, S);
  input A; input B; input CI; output CO; output S;
  assign S = A ^ B ^ CI; assign CO = (A & B) | (CI & (A ^ B));
endmodule

module HADDX1(A0, B0, C1, SO);
  input A0; input B0; output C1; output SO;
  assign SO = A0 ^ B0; assign C1 = A0 & B0;
endmodule

module INVX0(IN, QN);
  input IN; output QN;
  assign QN = ~IN;
endmodule

module INVX8(IN, QN);
  input IN; output QN;
  assign QN = ~IN;
endmodule

module ISOLANDX1(D, ISO, Q);
  input D; input ISO; output Q;
  assign Q = D & ISO;
endmodule

module LARX1(CLK, D, RSTB, Q, QN);
  input CLK; input D; input RSTB; output reg Q; output QN;
  always_latch begin if (!RSTB) Q = 1'b0; else Q = D; end assign QN = ~Q;
endmodule

module LSDNENX1(D, ENB, Q);
  input D; input ENB; output reg Q;
  always_latch begin if (!ENB) Q = D; end
endmodule

module LSDNX1(D, Q);
  input D; output reg Q;
  always_latch begin Q = D; end
endmodule

module MUX21X1(IN1, IN2, S, Q);
  input IN1; input IN2; input S; output Q;
  assign Q = S ? IN2 : IN1;
endmodule

module MUX21X2(IN1, IN2, S, Q);
  input IN1; input IN2; input S; output Q;
  assign Q = S ? IN2 : IN1;
endmodule

module MUX41X1(IN1, IN2, IN3, IN4, S0, S1, Q);
  input IN1; input IN2; input IN3; input IN4; input S0; input S1; output Q;
  assign Q = S1 ? (S0 ? IN4 : IN3) : (S0 ? IN2 : IN1);
endmodule

module NAND2X0(IN1, IN2, QN);
  input IN1; input IN2; output QN;
  assign QN = ~(IN1&IN2);
endmodule

module NAND2X1(IN1, IN2, QN);
  input IN1; input IN2; output QN;
  assign QN = ~(IN1&IN2);
endmodule

module NAND3X0(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~((IN1&IN2)&IN3);
endmodule

module NAND3X1(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~((IN1&IN2)&IN3);
endmodule

module NAND3X4(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~((IN1&IN2)&IN3);
endmodule

module NAND4X0(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~(((IN1&IN2)&IN3)&IN4);
endmodule

module NAND4X1(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~(((IN1&IN2)&IN3)&IN4);
endmodule

module NBUFFX2(IN, Q);
  input IN; output Q;
  assign Q = IN;
endmodule

module NOR2X0(IN1, IN2, QN);
  input IN1; input IN2; output QN;
  assign QN = ~(IN1|IN2);
endmodule

module NOR2X2(IN1, IN2, QN);
  input IN1; input IN2; output QN;
  assign QN = ~(IN1|IN2);
endmodule

module NOR3X0(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~((IN1|IN2)|IN3);
endmodule

module NOR3X1(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~((IN1|IN2)|IN3);
endmodule

module NOR4X0(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~(((IN1|IN2)|IN3)|IN4);
endmodule

module NOR4X1(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~(((IN1|IN2)|IN3)|IN4);
endmodule

module OA21X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = (IN1 | IN2) & IN3;
endmodule

module OA221X1(IN1, IN2, IN3, IN4, IN5, Q);
  input IN1; input IN2; input IN3; input IN4; input IN5; output Q;
  assign Q = (IN1 | IN2) & (IN3 | IN4) & IN5;
endmodule

module OA222X1(IN1, IN2, IN3, IN4, IN5, IN6, Q);
  input IN1; input IN2; input IN3; input IN4; input IN5; input IN6; output Q;
  assign Q = (IN1 | IN2) & (IN3 | IN4) & (IN5 | IN6);
endmodule

module OA22X1(IN1, IN2, IN3, IN4, Q);
  input IN1; input IN2; input IN3; input IN4; output Q;
  assign Q = (IN1 | IN2) & (IN3 | IN4);
endmodule

module OAI21X1(IN1, IN2, IN3, QN);
  input IN1; input IN2; input IN3; output QN;
  assign QN = ~(IN1&(IN2|IN3));
endmodule

module OAI221X1(IN1, IN2, IN3, IN4, IN5, QN);
  input IN1; input IN2; input IN3; input IN4; input IN5; output QN;
  assign QN = ~(((IN4|IN5)&IN1)&(IN2|IN3));
endmodule

module OAI222X1(IN1, IN2, IN3, IN4, IN5, IN6, QN);
  input IN1; input IN2; input IN3; input IN4; input IN5; input IN6; output QN;
  assign QN = ~(((IN1|IN2)&(IN3|IN4))&(IN5|IN6));
endmodule

module OAI22X1(IN1, IN2, IN3, IN4, QN);
  input IN1; input IN2; input IN3; input IN4; output QN;
  assign QN = ~((IN1|IN2)&(IN3|IN4));
endmodule

module OR2X1(IN1, IN2, Q);
  input IN1; input IN2; output Q;
  assign Q = (IN1|IN2);
endmodule

module OR3X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = ((IN1|IN2)|IN3);
endmodule

module OR4X1(IN1, IN2, IN3, IN4, Q);
  input IN1; input IN2; input IN3; input IN4; output Q;
  assign Q = (((IN1|IN2)|IN3)|IN4);
endmodule

module SDFFARX1(CLK, D, RSTB, SE, SI, Q, QN);
  input CLK; input D; input RSTB; input SE; input SI; output reg Q; output QN;
  always @(posedge CLK or negedge RSTB) if (!RSTB) Q <= 1'b0; else Q <= SE ? SI : D; assign QN = ~Q;
endmodule

module SDFFX1(CLK, D, SE, SI, Q, QN);
  input CLK; input D; input SE; input SI; output reg Q; output QN;
  always @(posedge CLK) Q <= SE ? SI : D; assign QN = ~Q;
endmodule

module XNOR2X1(IN1, IN2, Q);
  input IN1; input IN2; output Q;
  assign Q = ~(IN1^IN2);
endmodule

module XNOR3X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = ~(IN1 ^ IN2 ^ IN3);
endmodule

module XOR2X1(IN1, IN2, Q);
  input IN1; input IN2; output Q;
  assign Q = (IN1^IN2);
endmodule

module XOR3X1(IN1, IN2, IN3, Q);
  input IN1; input IN2; input IN3; output Q;
  assign Q = IN1 ^ IN2 ^ IN3;
endmodule
