/*
Copyright (c) 2012 Ben Croston

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "c_gpio.h"

#define BCM2708_PERI_BASE  0x20000000
#define GPIO_BASE          (BCM2708_PERI_BASE + 0x200000)
#define CLOCK_BASE         (BCM2708_PERI_BASE + 0x101000) 
#define FSEL_OFFSET        0   // 0x0000
#define SET_OFFSET         7   // 0x001c / 4
#define CLR_OFFSET         10  // 0x0028 / 4
#define PINLEVEL_OFFSET    13  // 0x0034 / 4
#define PULLUPDN_OFFSET    37  // 0x0094 / 4
#define PULLUPDNCLK_OFFSET 38  // 0x0098 / 4

/********************************************************\
 * General Function:
 * 
 * FSELn FSEL9n - Function Select n
 *    000 = GPIO Pin n is an input 
 *    001 = GPIO Pin n is an output 
 *    100 = GPIO Pin n takes alternate function 0 
 *    101 = GPIO Pin n takes alternate function 1 
 *    110 = GPIO Pin n takes alternate function 2 
 *    111 = GPIO Pin n takes alternate function 3 
 *    011 = GPIO Pin n takes alternate function 4 
 *    010 = GPIO Pin n takes alternate function 5 
 * 
 * #define SET_GPIO_ALT(g,a) *(gpio_map+(((g)/10))) |= (((a)<=3?(a)+4:(a)==4?3:2)<<(((g)%10)*3))
\******************************************************/
#define GP_CLK0_CTL *(clk_map + 0x1C)   // 4*0x1C = 0x70
#define GP_CLK0_DIV *(clk_map + 0x1D)	// 4*0x1D = 0x74
#define OSCILLATOR_FREQ 19000000 //Hz

#define PAGE_SIZE  (4*1024)
#define BLOCK_SIZE (4*1024)

static volatile uint32_t *gpio_map;
static volatile uint32_t *clk_map;

void short_wait(void)
{
    int i;
    
    for (i=0; i<100; i++)
    {
        i++;
        i--;
    }
}

int setup(void)
{
    int mem_fd;
    uint8_t *gpio_mem;
	uint8_t *clk_mem;
	
    if ((mem_fd = open("/dev/mem", O_RDWR|O_SYNC) ) < 0)
    {
        return SETUP_DEVMEM_FAIL;
    }
	
	// Allocate MAP block
    if ((gpio_mem = malloc(BLOCK_SIZE + (PAGE_SIZE-1))) == NULL)
        return SETUP_MALLOC_FAIL;
    if ((clk_mem = malloc(BLOCK_SIZE + (PAGE_SIZE-1))) == NULL)
		return SETUP_MALLOC_FAIL;

	// Make sure pointer is on 4K boundary
    if ((uint32_t)gpio_mem % PAGE_SIZE)
        gpio_mem += PAGE_SIZE - ((uint32_t)gpio_mem % PAGE_SIZE);
    if ((uint32_t)clk_mem % PAGE_SIZE)
		clk_mem += PAGE_SIZE - ((uint32_t)clk_mem % PAGE_SIZE);
        
	// Now map it
    gpio_map = (uint32_t *)mmap( (caddr_t)gpio_mem, BLOCK_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FIXED, mem_fd, GPIO_BASE);
	clk_map = (uint32_t *)mmap( (caddr_t)clk_mem, BLOCK_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FIXED, mem_fd, CLOCK_BASE);
	
    if ((uint32_t)gpio_map < 0)
        return SETUP_MMAP_FAIL;
	if ((uint32_t)clk_map < 0)
		return SETUP_MMAP_FAIL;
		
    return SETUP_OK;
}

void set_pullupdn(int gpio, int pud)
{
    int clk_offset = PULLUPDNCLK_OFFSET + (gpio/32);
    int shift = (gpio%32);
    
    if (pud == PUD_DOWN)
       *(gpio_map+PULLUPDN_OFFSET) = (*(gpio_map+PULLUPDN_OFFSET) & ~3) | PUD_DOWN;
    else if (pud == PUD_UP)
       *(gpio_map+PULLUPDN_OFFSET) = (*(gpio_map+PULLUPDN_OFFSET) & ~3) | PUD_UP;
    else  // pud == PUD_OFF
       *(gpio_map+PULLUPDN_OFFSET) &= ~3;
    
    short_wait();
    *(gpio_map+clk_offset) = 1 << shift;
    short_wait();
    *(gpio_map+PULLUPDN_OFFSET) &= ~3;
    *(gpio_map+clk_offset) = 0;
}

void setup_gpio(int gpio, int direction, int pud)
{
	//gpio here uses Broadcom numbering
	
    int offset = FSEL_OFFSET + (gpio/10);
    int shift = (gpio%10)*3;

    set_pullupdn(gpio, pud);
    if (direction == OUTPUT)
        *(gpio_map+offset) = (*(gpio_map+offset) & ~(7<<shift)) | (1<<shift);
    else if (direction == INPUT)
        *(gpio_map+offset) = (*(gpio_map+offset) & ~(7<<shift));
    else if (direction == ALT0 && gpio == 4){
		  //for now, limit this to the GPCLK0 pin as all others are unhandled anyway
		  //Set ENAB bit to 0 to stop clock and set source to oscillator:
		GP_CLK0_CTL = 0x5A000001;
		  //Set ALT0 function of pin:
		*(gpio_map+offset) = (*(gpio_map+offset) & ~(7<<shift)) | (4)<<shift;
		
	}
}

void output_gpio(int gpio, int value)
{
    int offset, shift;
    
    if (value) // value == HIGH
        offset = SET_OFFSET + (gpio/32);
    else       // value == LOW
        offset = CLR_OFFSET + (gpio/32);
    
    shift = (gpio%32);

    *(gpio_map+offset) = 1 << shift;
}

/**************************************************************\
 * 
 * GP_CLK0_CTL is a 32-bit word:
 * 	32-24     23-11  10-9    8     7     6  5     4     3-0	
 * 	[PASSWD]  [-]    [MASH]  FLIP  BUSY  -  KILL  ENAB  [SRC]
 * 
 * 	[PASSWD] = 0x5A
 *  [MASH] = 00 for integer division
 * 	[SRC] = 0001 for oscillator (at 15MHz?)
 * 	see Broadcom "BCM2835 ARM Peripherals" datasheet p107
 * 
 * GP_CLK0_DIV is a 32-bit word:
 *	31-24		23-12	11-0
 * 	[PASSWD]	[DIVI]	[DIVF]
 * 
 * 	[DIVI] = integer part of divisor
 * 	[DIVF] = fractional part of divisor
 *	see datasheet p108
 * 
\***************************************************************/
	
void clock_frequency(int gpio, int frequency)
{
		int divi,divf,prev;
		prev=0;
		
		divi = OSCILLATOR_FREQ/frequency;
		
		//We only have 12 bits to play with:
		//if (divi>8192)
		//	divi=8192;
		//put this check in py_gpio.c so that we can raise an exception
		
		divf = ((OSCILLATOR_FREQ % frequency)*4096)/frequency;
		
		//remember previous run state of the clock:
		if((GP_CLK0_CTL & 0x00000010) == 0x10)
			prev=1;
			
		//stop clock:
		GP_CLK0_CTL = 0x5A000001;
		usleep(10);
		
		//update divisor:
		//printf("Set DIVI= %d and DIVF= %d  : ", divi, divf);
		
		GP_CLK0_DIV = 0x5A000000 | (divi & 0xFFF)<<12 | (divf&0xFFF);
		
		//printf("GP_CLK0_DIV is now %#x\n", GP_CLK0_DIV);
		usleep(10);
		
		//start clock if it was previously running:
		if (prev==1)
			GP_CLK0_CTL = 0x5A000011;
}

void clock_toggle(int gpio, int enab)
{
	//printf("Toggling clock ENAB to %d  :  ", ((enab!=0)?1:0));
	
	GP_CLK0_CTL = 0x5A000001 | ((enab!=0)?1:0)<<4;
	
	//printf("GP_CLK0_CTL is now %#x\n", GP_CLK0_CTL);
}

int input_gpio(int gpio)
{
   int offset, value, mask;
   
   offset = PINLEVEL_OFFSET + (gpio/32);
   mask = (1 << gpio%32);
   value = *(gpio_map+offset) & mask;
   return value;
}

void cleanup(void)
{
    // fixme - set all gpios back to input
    munmap((caddr_t)gpio_map, BLOCK_SIZE);
}
