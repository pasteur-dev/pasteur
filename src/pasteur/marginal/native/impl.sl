	.file	"impl.c"
	.text
.Ltext0:
	.file 0 "/mnt/ext/projects/pasteur/src/pasteur/marginal/native" "impl.c"
	.type	sum_inline_u16, @function
sum_inline_u16:
.LFB4204:
	.file 1 "impl.c"
	.loc 1 103 1
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$1664, %rsp
	movl	%edi, 76(%rsp)
	movl	%esi, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movl	%ecx, 60(%rsp)
	movq	%r8, 48(%rsp)
	movq	%r9, 40(%rsp)
	movq	24(%rbp), %rax
	movq	%rax, 32(%rsp)
	movq	32(%rbp), %rax
	movq	%rax, 24(%rsp)
	.loc 1 103 1
	movq	%fs:40, %rax
	movq	%rax, 1656(%rsp)
	xorl	%eax, %eax
.LBB49:
	.loc 1 105 14
	movl	$0, 84(%rsp)
	.loc 1 105 5
	jmp	.L2
.L5:
.LBB50:
	.loc 1 107 18
	movl	$0, 88(%rsp)
	.loc 1 107 9
	jmp	.L3
.L4:
	.loc 1 109 52 discriminator 3
	movl	84(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	48(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 109 23 discriminator 3
	movl	84(%rsp), %eax
	sall	$4, %eax
	movl	%eax, %ecx
	.loc 1 109 29 discriminator 3
	movl	88(%rsp), %eax
	addl	%ecx, %eax
	.loc 1 109 34 discriminator 3
	cltq
	movw	%dx, 832(%rsp,%rax,2)
	.loc 1 107 39 discriminator 3
	incl	88(%rsp)
.L3:
	.loc 1 107 27 discriminator 1
	cmpl	$15, 88(%rsp)
	jle	.L4
.LBE50:
	.loc 1 105 33 discriminator 2
	incl	84(%rsp)
.L2:
	.loc 1 105 23 discriminator 1
	movl	84(%rsp), %eax
	cmpl	60(%rsp), %eax
	jl	.L5
.LBE49:
.LBB51:
	.loc 1 113 14
	movl	$0, 92(%rsp)
	.loc 1 113 5
	jmp	.L6
.L9:
.LBB52:
	.loc 1 115 18
	movl	$0, 96(%rsp)
	.loc 1 115 9
	jmp	.L7
.L8:
	.loc 1 117 54 discriminator 3
	movl	92(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	32(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 117 24 discriminator 3
	movl	92(%rsp), %eax
	sall	$4, %eax
	movl	%eax, %ecx
	.loc 1 117 30 discriminator 3
	movl	96(%rsp), %eax
	addl	%ecx, %eax
	.loc 1 117 35 discriminator 3
	cltq
	movw	%dx, 1248(%rsp,%rax,2)
	.loc 1 115 39 discriminator 3
	incl	96(%rsp)
.L7:
	.loc 1 115 27 discriminator 1
	cmpl	$15, 96(%rsp)
	jle	.L8
.LBE52:
	.loc 1 113 34 discriminator 2
	incl	92(%rsp)
.L6:
	.loc 1 113 23 discriminator 1
	movl	92(%rsp), %eax
	cmpl	16(%rbp), %eax
	jl	.L9
.LBE51:
	.loc 1 121 30
	movl	76(%rsp), %eax
	sall	$2, %eax
	.loc 1 121 19
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, 120(%rsp)
	.loc 1 122 15
	movq	120(%rsp), %rax
	movq	%rax, 128(%rsp)
	.loc 1 123 41
	movl	76(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	.loc 1 123 15
	movq	120(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 136(%rsp)
	.loc 1 124 45
	movl	76(%rsp), %eax
	addl	%eax, %eax
	cltq
	.loc 1 124 41
	leaq	0(,%rax,4), %rdx
	.loc 1 124 15
	movq	120(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 144(%rsp)
	.loc 1 125 45
	movl	76(%rsp), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	cltq
	.loc 1 125 41
	leaq	0(,%rax,4), %rdx
	.loc 1 125 15
	movq	120(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 152(%rsp)
.LBB53:
	.loc 1 127 14
	movl	$0, 100(%rsp)
	.loc 1 127 5
	jmp	.L10
.L11:
	.loc 1 128 14 discriminator 3
	movl	100(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	.loc 1 128 18 discriminator 3
	movl	$0, (%rax)
	.loc 1 127 35 discriminator 3
	incl	100(%rsp)
.L10:
	.loc 1 127 27 discriminator 1
	movl	76(%rsp), %eax
	sall	$2, %eax
	.loc 1 127 23 discriminator 1
	cmpl	%eax, 100(%rsp)
	jl	.L11
.LBE53:
.LBB54:
	.loc 1 130 14
	movl	$0, 104(%rsp)
	.loc 1 130 5
	jmp	.L12
.L27:
.LBB55:
.LBB56:
.LBB57:
	.file 2 "/usr/lib/gcc/x86_64-linux-gnu/11/include/avxintrin.h"
	.loc 2 1244 24
	vpxor	%xmm0, %xmm0, %xmm0
.LBE57:
.LBE56:
	.loc 1 133 23
	vmovdqa	%ymm0, 480(%rsp)
.LBB58:
	.loc 1 135 18
	movl	$0, 108(%rsp)
	.loc 1 135 9
	jmp	.L14
.L20:
.LBB59:
	.loc 1 137 62 discriminator 3
	movl	108(%rsp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	40(%rsp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	.loc 1 137 28 discriminator 3
	movl	104(%rsp), %eax
	cltq
	.loc 1 137 55 discriminator 3
	addq	%rdx, %rax
	movq	%rax, 168(%rsp)
.LBB60:
.LBB61:
	.file 3 "/usr/lib/gcc/x86_64-linux-gnu/11/include/pmmintrin.h"
	.loc 3 113 20 discriminator 3
	movq	168(%rsp), %rax
	vlddqu	(%rax), %xmm0
.LBE61:
.LBE60:
	.loc 1 137 28 discriminator 3
	vmovdqa	%xmm0, 448(%rsp)
	vmovdqa	448(%rsp), %xmm0
	vmovdqa	%xmm0, 464(%rsp)
.LBB62:
.LBB63:
	.file 4 "/usr/lib/gcc/x86_64-linux-gnu/11/include/avx2intrin.h"
	.loc 4 484 20 discriminator 3
	vmovdqa	464(%rsp), %xmm0
	vpmovzxbw	%xmm0, %ymm0
.LBE63:
.LBE62:
	.loc 1 138 19 discriminator 3
	vmovdqa	%ymm0, 512(%rsp)
	.loc 1 139 58 discriminator 3
	movl	108(%rsp), %eax
	sall	$4, %eax
	.loc 1 139 48 discriminator 3
	leaq	832(%rsp), %rdx
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movq	%rax, 160(%rsp)
.LBB64:
.LBB65:
	.loc 2 917 10 discriminator 3
	movq	160(%rsp), %rax
	vmovdqa	(%rax), %ymm0
.LBE65:
.LBE64:
	.loc 1 139 19 discriminator 3
	vmovdqa	%ymm0, 544(%rsp)
	vmovdqa	512(%rsp), %ymm0
	vmovdqa	%ymm0, 640(%rsp)
	vmovdqa	544(%rsp), %ymm0
	vmovdqa	%ymm0, 672(%rsp)
.LBB66:
.LBB67:
	.loc 4 555 21 discriminator 3
	vmovdqa	640(%rsp), %ymm1
	.loc 4 555 36 discriminator 3
	vmovdqa	672(%rsp), %ymm0
	.loc 4 555 34 discriminator 3
	vpmullw	%ymm0, %ymm1, %ymm0
.LBE67:
.LBE66:
	.loc 1 140 19 discriminator 3
	vmovdqa	%ymm0, 512(%rsp)
	vmovdqa	512(%rsp), %ymm0
	vmovdqa	%ymm0, 576(%rsp)
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%ymm0, 608(%rsp)
.LBB68:
.LBB69:
	.loc 4 156 19 discriminator 3
	vmovdqa	608(%rsp), %ymm0
	vmovdqa	576(%rsp), %ymm1
	vpaddusw	%ymm0, %ymm1, %ymm0
	.loc 4 156 10 discriminator 3
	nop
.LBE69:
.LBE68:
	.loc 1 141 19 discriminator 3
	vmovdqa	%ymm0, 480(%rsp)
.LBE59:
	.loc 1 135 37 discriminator 3
	incl	108(%rsp)
.L14:
	.loc 1 135 27 discriminator 1
	movl	108(%rsp), %eax
	cmpl	60(%rsp), %eax
	jl	.L20
.LBE58:
.LBB70:
	.loc 1 144 18
	movl	$0, 112(%rsp)
	.loc 1 144 9
	jmp	.L21
.L26:
	.loc 1 146 57 discriminator 3
	movl	112(%rsp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	24(%rsp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	.loc 1 146 60 discriminator 3
	movl	104(%rsp), %eax
	cltq
	addq	%rax, %rax
	.loc 1 146 49 discriminator 3
	addq	%rdx, %rax
	movq	%rax, 184(%rsp)
.LBB71:
.LBB72:
	.loc 2 1011 20 discriminator 3
	movq	184(%rsp), %rax
	vlddqu	(%rax), %ymm0
.LBE72:
.LBE71:
	.loc 1 146 19 discriminator 3
	vmovdqa	%ymm0, 512(%rsp)
	.loc 1 147 59 discriminator 3
	movl	112(%rsp), %eax
	sall	$4, %eax
	.loc 1 147 48 discriminator 3
	leaq	1248(%rsp), %rdx
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movq	%rax, 176(%rsp)
.LBB73:
.LBB74:
	.loc 2 917 10 discriminator 3
	movq	176(%rsp), %rax
	vmovdqa	(%rax), %ymm0
.LBE74:
.LBE73:
	.loc 1 147 19 discriminator 3
	vmovdqa	%ymm0, 544(%rsp)
	vmovdqa	512(%rsp), %ymm0
	vmovdqa	%ymm0, 768(%rsp)
	vmovdqa	544(%rsp), %ymm0
	vmovdqa	%ymm0, 800(%rsp)
.LBB75:
.LBB76:
	.loc 4 555 21 discriminator 3
	vmovdqa	768(%rsp), %ymm1
	.loc 4 555 36 discriminator 3
	vmovdqa	800(%rsp), %ymm0
	.loc 4 555 34 discriminator 3
	vpmullw	%ymm0, %ymm1, %ymm0
.LBE76:
.LBE75:
	.loc 1 148 19 discriminator 3
	vmovdqa	%ymm0, 512(%rsp)
	vmovdqa	512(%rsp), %ymm0
	vmovdqa	%ymm0, 704(%rsp)
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%ymm0, 736(%rsp)
.LBB77:
.LBB78:
	.loc 4 156 19 discriminator 3
	vmovdqa	736(%rsp), %ymm0
	vmovdqa	704(%rsp), %ymm1
	vpaddusw	%ymm0, %ymm1, %ymm0
	.loc 4 156 10 discriminator 3
	nop
.LBE78:
.LBE77:
	.loc 1 149 19 discriminator 3
	vmovdqa	%ymm0, 480(%rsp)
	.loc 1 144 38 discriminator 3
	incl	112(%rsp)
.L21:
	.loc 1 144 27 discriminator 1
	movl	112(%rsp), %eax
	cmpl	16(%rbp), %eax
	jl	.L26
.LBE70:
.LBB79:
	.loc 1 153 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 192(%rsp)
	vmovdqa	192(%rsp), %xmm0
	vpextrw	$0, %xmm0, %eax
	movzwl	%ax, %eax
.LBE79:
	.loc 1 153 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 154 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	128(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB80:
	.loc 1 155 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 208(%rsp)
	vmovdqa	208(%rsp), %xmm0
	vpextrw	$1, %xmm0, %eax
	movzwl	%ax, %eax
.LBE80:
	.loc 1 155 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 156 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	136(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	136(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB81:
	.loc 1 157 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 224(%rsp)
	vmovdqa	224(%rsp), %xmm0
	vpextrw	$2, %xmm0, %eax
	movzwl	%ax, %eax
.LBE81:
	.loc 1 157 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 158 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	144(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB82:
	.loc 1 159 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 240(%rsp)
	vmovdqa	240(%rsp), %xmm0
	vpextrw	$3, %xmm0, %eax
	movzwl	%ax, %eax
.LBE82:
	.loc 1 159 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 160 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	152(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	152(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB83:
	.loc 1 161 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 256(%rsp)
	vmovdqa	256(%rsp), %xmm0
	vpextrw	$4, %xmm0, %eax
	movzwl	%ax, %eax
.LBE83:
	.loc 1 161 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 162 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	128(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB84:
	.loc 1 163 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 272(%rsp)
	vmovdqa	272(%rsp), %xmm0
	vpextrw	$5, %xmm0, %eax
	movzwl	%ax, %eax
.LBE84:
	.loc 1 163 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 164 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	136(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	136(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB85:
	.loc 1 165 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 288(%rsp)
	vmovdqa	288(%rsp), %xmm0
	vpextrw	$6, %xmm0, %eax
	movzwl	%ax, %eax
.LBE85:
	.loc 1 165 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 166 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	144(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB86:
	.loc 1 167 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vmovdqa	%xmm0, 304(%rsp)
	vmovdqa	304(%rsp), %xmm0
	vpextrw	$7, %xmm0, %eax
	movzwl	%ax, %eax
.LBE86:
	.loc 1 167 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 168 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	152(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	152(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB87:
	.loc 1 169 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 320(%rsp)
	vmovdqa	320(%rsp), %xmm0
	vpextrw	$0, %xmm0, %eax
	movzwl	%ax, %eax
.LBE87:
	.loc 1 169 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 170 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	128(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB88:
	.loc 1 171 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 336(%rsp)
	vmovdqa	336(%rsp), %xmm0
	vpextrw	$1, %xmm0, %eax
	movzwl	%ax, %eax
.LBE88:
	.loc 1 171 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 172 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	136(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	136(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB89:
	.loc 1 173 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 352(%rsp)
	vmovdqa	352(%rsp), %xmm0
	vpextrw	$2, %xmm0, %eax
	movzwl	%ax, %eax
.LBE89:
	.loc 1 173 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 174 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	144(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB90:
	.loc 1 175 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 368(%rsp)
	vmovdqa	368(%rsp), %xmm0
	vpextrw	$3, %xmm0, %eax
	movzwl	%ax, %eax
.LBE90:
	.loc 1 175 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 176 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	152(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	152(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB91:
	.loc 1 177 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 384(%rsp)
	vmovdqa	384(%rsp), %xmm0
	vpextrw	$4, %xmm0, %eax
	movzwl	%ax, %eax
.LBE91:
	.loc 1 177 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 178 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	128(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB92:
	.loc 1 179 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 400(%rsp)
	vmovdqa	400(%rsp), %xmm0
	vpextrw	$5, %xmm0, %eax
	movzwl	%ax, %eax
.LBE92:
	.loc 1 179 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 180 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	136(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	136(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB93:
	.loc 1 181 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 416(%rsp)
	vmovdqa	416(%rsp), %xmm0
	vpextrw	$6, %xmm0, %eax
	movzwl	%ax, %eax
.LBE93:
	.loc 1 181 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 182 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	144(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBB94:
	.loc 1 183 13 discriminator 2
	vmovdqa	480(%rsp), %ymm0
	vextracti128	$0x1, %ymm0, %xmm0
	vmovdqa	%xmm0, 432(%rsp)
	vmovdqa	432(%rsp), %xmm0
	vpextrw	$7, %xmm0, %eax
	movzwl	%ax, %eax
.LBE94:
	.loc 1 183 11 discriminator 2
	movw	%ax, 82(%rsp)
	.loc 1 184 18 discriminator 2
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rdx
	movq	152(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movzwl	82(%rsp), %eax
	leaq	0(,%rax,4), %rcx
	movq	152(%rsp), %rax
	addq	%rcx, %rax
	incl	%edx
	movl	%edx, (%rax)
.LBE55:
	.loc 1 130 35 discriminator 2
	addl	$16, 104(%rsp)
.L12:
	.loc 1 130 27 discriminator 1
	movl	72(%rsp), %eax
	subl	$15, %eax
	.loc 1 130 23 discriminator 1
	cmpl	%eax, 104(%rsp)
	jl	.L27
.LBE54:
.LBB95:
	.loc 1 187 14
	movl	$0, 116(%rsp)
	.loc 1 187 5
	jmp	.L28
.L29:
	.loc 1 189 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	.loc 1 189 26 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	136(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 189 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	128(%rsp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	.loc 1 190 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	.loc 1 190 26 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	152(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 190 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	144(%rsp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	.loc 1 191 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	.loc 1 191 26 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	144(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 191 18 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	128(%rsp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	.loc 1 192 16 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	64(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	.loc 1 192 24 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	128(%rsp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	.loc 1 192 16 discriminator 3
	movl	116(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rsi
	movq	64(%rsp), %rax
	addq	%rsi, %rax
	addl	%ecx, %edx
	movl	%edx, (%rax)
	.loc 1 187 31 discriminator 3
	incl	116(%rsp)
.L28:
	.loc 1 187 23 discriminator 1
	movl	116(%rsp), %eax
	cmpl	76(%rsp), %eax
	jl	.L29
.LBE95:
	.loc 1 195 5
	movq	120(%rsp), %rax
	movq	%rax, %rdi
	call	free@PLT
	.loc 1 196 1
	nop
	movq	1656(%rsp), %rax
	subq	%fs:40, %rax
	je	.L30
	call	__stack_chk_fail@PLT
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4204:
	.size	sum_inline_u16, .-sum_inline_u16
	.local	out
	.comm	out,40000000,32
	.local	a1
	.comm	a1,10000000,32
	.local	a2
	.comm	a2,20000000,32
	.section	.rodata
.LC0:
	.string	"%d "
	.text
	.globl	main
	.type	main, @function
main:
.LFB4205:
	.loc 1 311 1
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	.loc 1 311 1
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
.LBB96:
	.loc 1 312 14
	movl	$0, -44(%rbp)
	.loc 1 312 5
	jmp	.L32
.L33:
	.loc 1 314 19 discriminator 3
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2139062143, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$7, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	movl	%edx, %ecx
	sall	$8, %ecx
	subl	%edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	.loc 1 314 15 discriminator 3
	movl	%edx, %ecx
	movl	-44(%rbp), %eax
	cltq
	leaq	a1(%rip), %rdx
	movb	%cl, (%rax,%rdx)
	.loc 1 315 20 discriminator 3
	movl	-44(%rbp), %eax
	leal	5(%rax), %edx
	.loc 1 315 25 discriminator 3
	movslq	%edx, %rax
	imulq	$-2139062143, %rax, %rax
	shrq	$32, %rax
	addl	%edx, %eax
	sarl	$7, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$8, %ecx
	subl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	.loc 1 315 15 discriminator 3
	movl	%eax, %ecx
	movl	-44(%rbp), %eax
	cltq
	addq	%rax, %rax
	leaq	a2(%rip), %rdx
	movw	%cx, (%rax,%rdx)
	.loc 1 316 16 discriminator 3
	movl	-44(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	out(%rip), %rdx
	movl	$0, (%rax,%rdx)
	.loc 1 312 29 discriminator 3
	incl	-44(%rbp)
.L32:
	.loc 1 312 23 discriminator 1
	cmpl	$9999999, -44(%rbp)
	jle	.L33
.LBE96:
	.loc 1 319 9
	movl	$1, -32(%rbp)
	.loc 1 320 14
	leaq	a1(%rip), %rax
	movq	%rax, -24(%rbp)
	.loc 1 321 9
	movl	$256, -28(%rbp)
	.loc 1 322 15
	leaq	a2(%rip), %rax
	movq	%rax, -16(%rbp)
	.loc 1 324 5
	leaq	-24(%rbp), %rcx
	leaq	-32(%rbp), %rdx
	subq	$8, %rsp
	leaq	-16(%rbp), %rax
	pushq	%rax
	leaq	-28(%rbp), %rax
	pushq	%rax
	pushq	$1
	movq	%rcx, %r9
	movq	%rdx, %r8
	movl	$1, %ecx
	leaq	out(%rip), %rax
	movq	%rax, %rdx
	movl	$10000000, %esi
	movl	$10000, %edi
	call	sum_inline_u16
	addq	$32, %rsp
.LBB97:
	.loc 1 326 14
	movl	$0, -40(%rbp)
	.loc 1 326 5
	jmp	.L34
.L37:
.LBB98:
	.loc 1 328 18
	movl	$0, -36(%rbp)
	.loc 1 328 9
	jmp	.L35
.L36:
	.loc 1 330 35 discriminator 3
	movl	-40(%rbp), %eax
	sall	$8, %eax
	movl	%eax, %edx
	.loc 1 330 39 discriminator 3
	movl	-36(%rbp), %eax
	addl	%edx, %eax
	.loc 1 330 13 discriminator 3
	cltq
	salq	$2, %rax
	leaq	out(%rip), %rdx
	movl	(%rax,%rdx), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	.loc 1 328 34 discriminator 3
	incl	-36(%rbp)
.L35:
	.loc 1 328 27 discriminator 1
	cmpl	$31, -36(%rbp)
	jle	.L36
.LBE98:
	.loc 1 332 9 discriminator 2
	movl	$10, %edi
	call	putchar@PLT
	.loc 1 326 30 discriminator 2
	incl	-40(%rbp)
.L34:
	.loc 1 326 23 discriminator 1
	cmpl	$31, -40(%rbp)
	jle	.L37
.LBE97:
	.loc 1 334 12
	movl	$1, %eax
	.loc 1 335 1
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L39
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4205:
	.size	main, .-main
.Letext0:
	.file 5 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 6 "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h"
	.file 7 "/usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h"
	.file 8 "/usr/lib/gcc/x86_64-linux-gnu/11/include/emmintrin.h"
	.file 9 "/usr/include/stdlib.h"
	.file 10 "/usr/include/stdio.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0xaf9
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x1e
	.long	.LASF54
	.byte	0x1d
	.long	.LASF0
	.long	.LASF1
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.long	.Ldebug_line0
	.uleb128 0x4
	.byte	0x1
	.byte	0x8
	.long	.LASF2
	.uleb128 0x4
	.byte	0x2
	.byte	0x7
	.long	.LASF3
	.uleb128 0x4
	.byte	0x4
	.byte	0x7
	.long	.LASF4
	.uleb128 0x4
	.byte	0x8
	.byte	0x7
	.long	.LASF5
	.uleb128 0x4
	.byte	0x1
	.byte	0x6
	.long	.LASF6
	.uleb128 0x3
	.long	.LASF8
	.byte	0x5
	.byte	0x26
	.byte	0x17
	.long	0x2e
	.uleb128 0x4
	.byte	0x2
	.byte	0x5
	.long	.LASF7
	.uleb128 0x3
	.long	.LASF9
	.byte	0x5
	.byte	0x28
	.byte	0x1c
	.long	0x35
	.uleb128 0x1f
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x3
	.long	.LASF10
	.byte	0x5
	.byte	0x2a
	.byte	0x16
	.long	0x3c
	.uleb128 0x4
	.byte	0x8
	.byte	0x5
	.long	.LASF11
	.uleb128 0x20
	.byte	0x8
	.uleb128 0x6
	.long	0x91
	.uleb128 0x4
	.byte	0x1
	.byte	0x6
	.long	.LASF12
	.uleb128 0x12
	.long	0x91
	.uleb128 0x3
	.long	.LASF13
	.byte	0x6
	.byte	0x18
	.byte	0x13
	.long	0x51
	.uleb128 0x3
	.long	.LASF14
	.byte	0x6
	.byte	0x19
	.byte	0x14
	.long	0x64
	.uleb128 0x3
	.long	.LASF15
	.byte	0x6
	.byte	0x1a
	.byte	0x14
	.long	0x77
	.uleb128 0x3
	.long	.LASF16
	.byte	0x7
	.byte	0xd1
	.byte	0x17
	.long	0x43
	.uleb128 0x4
	.byte	0x8
	.byte	0x5
	.long	.LASF17
	.uleb128 0x4
	.byte	0x10
	.byte	0x4
	.long	.LASF18
	.uleb128 0x4
	.byte	0x8
	.byte	0x7
	.long	.LASF19
	.uleb128 0x4
	.byte	0x4
	.byte	0x4
	.long	.LASF20
	.uleb128 0x4
	.byte	0x8
	.byte	0x4
	.long	.LASF21
	.uleb128 0x3
	.long	.LASF22
	.byte	0x8
	.byte	0x2d
	.byte	0xf
	.long	0xfc
	.uleb128 0x7
	.long	0x5d
	.long	0x108
	.uleb128 0x8
	.byte	0x7
	.byte	0
	.uleb128 0x3
	.long	.LASF23
	.byte	0x8
	.byte	0x2f
	.byte	0xe
	.long	0x114
	.uleb128 0x7
	.long	0x91
	.long	0x120
	.uleb128 0x8
	.byte	0xf
	.byte	0
	.uleb128 0x3
	.long	.LASF24
	.byte	0x8
	.byte	0x35
	.byte	0x13
	.long	0x131
	.uleb128 0x12
	.long	0x120
	.uleb128 0x7
	.long	0xcd
	.long	0x13d
	.uleb128 0x8
	.byte	0x1
	.byte	0
	.uleb128 0x3
	.long	.LASF25
	.byte	0x2
	.byte	0x2b
	.byte	0x13
	.long	0x149
	.uleb128 0x7
	.long	0xcd
	.long	0x155
	.uleb128 0x8
	.byte	0x3
	.byte	0
	.uleb128 0x3
	.long	.LASF26
	.byte	0x2
	.byte	0x2d
	.byte	0xd
	.long	0x161
	.uleb128 0x7
	.long	0x70
	.long	0x16d
	.uleb128 0x8
	.byte	0x7
	.byte	0
	.uleb128 0x3
	.long	.LASF27
	.byte	0x2
	.byte	0x2f
	.byte	0xf
	.long	0x179
	.uleb128 0x7
	.long	0x5d
	.long	0x185
	.uleb128 0x8
	.byte	0xf
	.byte	0
	.uleb128 0x3
	.long	.LASF28
	.byte	0x2
	.byte	0x30
	.byte	0x18
	.long	0x191
	.uleb128 0x7
	.long	0x35
	.long	0x19d
	.uleb128 0x8
	.byte	0xf
	.byte	0
	.uleb128 0x3
	.long	.LASF29
	.byte	0x2
	.byte	0x39
	.byte	0x13
	.long	0x1ae
	.uleb128 0x12
	.long	0x19d
	.uleb128 0x7
	.long	0xcd
	.long	0x1ba
	.uleb128 0x8
	.byte	0x3
	.byte	0
	.uleb128 0x9
	.long	0xb5
	.long	0x1c9
	.uleb128 0x13
	.long	0x43
	.byte	0
	.uleb128 0x14
	.string	"out"
	.value	0x132
	.byte	0x11
	.long	0x1ba
	.uleb128 0x9
	.byte	0x3
	.quad	out
	.uleb128 0x9
	.long	0x9d
	.long	0x1ee
	.uleb128 0x13
	.long	0x43
	.byte	0
	.uleb128 0x14
	.string	"a1"
	.value	0x133
	.byte	0x10
	.long	0x1df
	.uleb128 0x9
	.byte	0x3
	.quad	a1
	.uleb128 0x9
	.long	0xa9
	.long	0x212
	.uleb128 0x13
	.long	0x43
	.byte	0
	.uleb128 0x14
	.string	"a2"
	.value	0x134
	.byte	0x11
	.long	0x203
	.uleb128 0x9
	.byte	0x3
	.quad	a2
	.uleb128 0x21
	.long	.LASF55
	.byte	0x9
	.value	0x22b
	.byte	0xd
	.long	0x23a
	.uleb128 0x15
	.long	0x8a
	.byte	0
	.uleb128 0x19
	.long	.LASF30
	.byte	0x9
	.value	0x21c
	.byte	0xe
	.long	0x8a
	.long	0x251
	.uleb128 0x15
	.long	0xc1
	.byte	0
	.uleb128 0x19
	.long	.LASF31
	.byte	0xa
	.value	0x164
	.byte	0xc
	.long	0x70
	.long	0x269
	.uleb128 0x15
	.long	0x269
	.uleb128 0x22
	.byte	0
	.uleb128 0x6
	.long	0x98
	.uleb128 0x23
	.long	.LASF56
	.byte	0x1
	.value	0x136
	.byte	0x5
	.long	0x70
	.quad	.LFB4205
	.quad	.LFE4205-.LFB4205
	.uleb128 0x1
	.byte	0x9c
	.long	0x34b
	.uleb128 0x1a
	.long	.LASF32
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x91
	.sleb128 -68
	.uleb128 0x1a
	.long	.LASF33
	.byte	0x1a
	.long	0x34b
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.uleb128 0xf
	.long	.LASF34
	.value	0x13f
	.byte	0x9
	.long	0x350
	.uleb128 0x2
	.byte	0x91
	.sleb128 -48
	.uleb128 0xf
	.long	.LASF35
	.value	0x140
	.byte	0xe
	.long	0x360
	.uleb128 0x2
	.byte	0x91
	.sleb128 -40
	.uleb128 0xf
	.long	.LASF36
	.value	0x141
	.byte	0x9
	.long	0x350
	.uleb128 0x2
	.byte	0x91
	.sleb128 -44
	.uleb128 0xf
	.long	.LASF37
	.value	0x142
	.byte	0xf
	.long	0x375
	.uleb128 0x2
	.byte	0x91
	.sleb128 -32
	.uleb128 0x2
	.quad	.LBB96
	.quad	.LBE96-.LBB96
	.long	0x30c
	.uleb128 0x16
	.string	"i"
	.value	0x138
	.byte	0xe
	.long	0x70
	.uleb128 0x2
	.byte	0x91
	.sleb128 -60
	.byte	0
	.uleb128 0xa
	.quad	.LBB97
	.quad	.LBE97-.LBB97
	.uleb128 0x16
	.string	"i"
	.value	0x146
	.byte	0xe
	.long	0x70
	.uleb128 0x2
	.byte	0x91
	.sleb128 -56
	.uleb128 0xa
	.quad	.LBB98
	.quad	.LBE98-.LBB98
	.uleb128 0x16
	.string	"j"
	.value	0x148
	.byte	0x12
	.long	0x70
	.uleb128 0x2
	.byte	0x91
	.sleb128 -52
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x6
	.long	0x8c
	.uleb128 0x9
	.long	0x70
	.long	0x360
	.uleb128 0x10
	.long	0x43
	.byte	0
	.byte	0
	.uleb128 0x9
	.long	0x370
	.long	0x370
	.uleb128 0x10
	.long	0x43
	.byte	0
	.byte	0
	.uleb128 0x6
	.long	0x9d
	.uleb128 0x9
	.long	0x385
	.long	0x385
	.uleb128 0x10
	.long	0x43
	.byte	0
	.byte	0
	.uleb128 0x6
	.long	0xa9
	.uleb128 0x24
	.long	.LASF57
	.byte	0x1
	.byte	0x63
	.byte	0x14
	.quad	.LFB4204
	.quad	.LFE4204-.LFB4204
	.uleb128 0x1
	.byte	0x9c
	.long	0x9f7
	.uleb128 0x17
	.string	"dom"
	.byte	0x9
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 76
	.uleb128 0x17
	.string	"l"
	.byte	0x12
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 72
	.uleb128 0x17
	.string	"out"
	.byte	0x1f
	.long	0x9f7
	.uleb128 0x3
	.byte	0x77
	.sleb128 64
	.uleb128 0xc
	.long	.LASF38
	.byte	0x65
	.byte	0x9
	.long	0x70
	.uleb128 0x2
	.byte	0x77
	.sleb128 60
	.uleb128 0xc
	.long	.LASF34
	.byte	0x65
	.byte	0x14
	.long	0x9fc
	.uleb128 0x2
	.byte	0x77
	.sleb128 48
	.uleb128 0xc
	.long	.LASF35
	.byte	0x65
	.byte	0x26
	.long	0xa01
	.uleb128 0x2
	.byte	0x77
	.sleb128 40
	.uleb128 0xc
	.long	.LASF39
	.byte	0x66
	.byte	0x9
	.long	0x70
	.uleb128 0x2
	.byte	0x91
	.sleb128 0
	.uleb128 0xc
	.long	.LASF36
	.byte	0x66
	.byte	0x15
	.long	0x9fc
	.uleb128 0x2
	.byte	0x77
	.sleb128 32
	.uleb128 0xc
	.long	.LASF37
	.byte	0x66
	.byte	0x29
	.long	0xa06
	.uleb128 0x2
	.byte	0x77
	.sleb128 24
	.uleb128 0x1b
	.long	.LASF40
	.byte	0x68
	.long	0xa0b
	.uleb128 0x3
	.byte	0x77
	.sleb128 832
	.uleb128 0x1b
	.long	.LASF41
	.byte	0x70
	.long	0xa0b
	.uleb128 0x3
	.byte	0x77
	.sleb128 1248
	.uleb128 0xd
	.long	.LASF42
	.byte	0x79
	.byte	0xb
	.long	0x8a
	.uleb128 0x3
	.byte	0x77
	.sleb128 120
	.uleb128 0xd
	.long	.LASF43
	.byte	0x7a
	.byte	0xf
	.long	0x9f7
	.uleb128 0x3
	.byte	0x77
	.sleb128 128
	.uleb128 0xd
	.long	.LASF44
	.byte	0x7b
	.byte	0xf
	.long	0x9f7
	.uleb128 0x3
	.byte	0x77
	.sleb128 136
	.uleb128 0xd
	.long	.LASF45
	.byte	0x7c
	.byte	0xf
	.long	0x9f7
	.uleb128 0x3
	.byte	0x77
	.sleb128 144
	.uleb128 0xd
	.long	.LASF46
	.byte	0x7d
	.byte	0xf
	.long	0x9f7
	.uleb128 0x3
	.byte	0x77
	.sleb128 152
	.uleb128 0x2
	.quad	.LBB49
	.quad	.LBE49-.LBB49
	.long	0x4cd
	.uleb128 0x1
	.string	"i"
	.byte	0x69
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 84
	.uleb128 0xa
	.quad	.LBB50
	.quad	.LBE50-.LBB50
	.uleb128 0x1
	.string	"j"
	.byte	0x6b
	.byte	0x12
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 88
	.byte	0
	.byte	0
	.uleb128 0x2
	.quad	.LBB51
	.quad	.LBE51-.LBB51
	.long	0x50f
	.uleb128 0x1
	.string	"i"
	.byte	0x71
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 92
	.uleb128 0xa
	.quad	.LBB52
	.quad	.LBE52-.LBB52
	.uleb128 0x1
	.string	"j"
	.byte	0x73
	.byte	0x12
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 96
	.byte	0
	.byte	0
	.uleb128 0x2
	.quad	.LBB53
	.quad	.LBE53-.LBB53
	.long	0x532
	.uleb128 0x1
	.string	"i"
	.byte	0x7f
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 100
	.byte	0
	.uleb128 0x2
	.quad	.LBB54
	.quad	.LBE54-.LBB54
	.long	0x9d7
	.uleb128 0x1
	.string	"i"
	.byte	0x82
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 104
	.uleb128 0xa
	.quad	.LBB55
	.quad	.LBE55-.LBB55
	.uleb128 0x1
	.string	"tmp"
	.byte	0x84
	.byte	0x11
	.long	0x19d
	.uleb128 0x3
	.byte	0x77
	.sleb128 512
	.uleb128 0x1
	.string	"mul"
	.byte	0x84
	.byte	0x16
	.long	0x19d
	.uleb128 0x3
	.byte	0x77
	.sleb128 544
	.uleb128 0x1
	.string	"idx"
	.byte	0x85
	.byte	0x11
	.long	0x19d
	.uleb128 0x3
	.byte	0x77
	.sleb128 480
	.uleb128 0x1
	.string	"k"
	.byte	0x98
	.byte	0x12
	.long	0xa9
	.uleb128 0x3
	.byte	0x77
	.sleb128 82
	.uleb128 0x2
	.quad	.LBB58
	.quad	.LBE58-.LBB58
	.long	0x6a9
	.uleb128 0x1
	.string	"j"
	.byte	0x87
	.byte	0x12
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 108
	.uleb128 0xa
	.quad	.LBB59
	.quad	.LBE59-.LBB59
	.uleb128 0xd
	.long	.LASF47
	.byte	0x89
	.byte	0x15
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 448
	.uleb128 0xb
	.long	0xadb
	.quad	.LBB60
	.quad	.LBE60-.LBB60
	.byte	0x89
	.byte	0x1c
	.long	0x606
	.uleb128 0x5
	.long	0xaea
	.uleb128 0x3
	.byte	0x77
	.sleb128 168
	.byte	0
	.uleb128 0xb
	.long	0xa46
	.quad	.LBB62
	.quad	.LBE62-.LBB62
	.byte	0x8a
	.byte	0x13
	.long	0x62b
	.uleb128 0x5
	.long	0xa56
	.uleb128 0x3
	.byte	0x77
	.sleb128 464
	.byte	0
	.uleb128 0xb
	.long	0xabd
	.quad	.LBB64
	.quad	.LBE64-.LBB64
	.byte	0x8b
	.byte	0x13
	.long	0x650
	.uleb128 0x5
	.long	0xacd
	.uleb128 0x3
	.byte	0x77
	.sleb128 160
	.byte	0
	.uleb128 0xb
	.long	0xa1b
	.quad	.LBB66
	.quad	.LBE66-.LBB66
	.byte	0x8c
	.byte	0x13
	.long	0x67e
	.uleb128 0x5
	.long	0xa38
	.uleb128 0x3
	.byte	0x77
	.sleb128 672
	.uleb128 0x5
	.long	0xa2b
	.uleb128 0x3
	.byte	0x77
	.sleb128 640
	.byte	0
	.uleb128 0x1c
	.long	0xa64
	.quad	.LBB68
	.quad	.LBE68-.LBB68
	.byte	0x8d
	.uleb128 0x5
	.long	0xa7f
	.uleb128 0x3
	.byte	0x77
	.sleb128 608
	.uleb128 0x5
	.long	0xa73
	.uleb128 0x3
	.byte	0x77
	.sleb128 576
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x2
	.quad	.LBB70
	.quad	.LBE70-.LBB70
	.long	0x76d
	.uleb128 0x1
	.string	"j"
	.byte	0x90
	.byte	0x12
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 112
	.uleb128 0xb
	.long	0xa9a
	.quad	.LBB71
	.quad	.LBE71-.LBB71
	.byte	0x92
	.byte	0x13
	.long	0x6f0
	.uleb128 0x5
	.long	0xaaa
	.uleb128 0x3
	.byte	0x77
	.sleb128 184
	.byte	0
	.uleb128 0xb
	.long	0xabd
	.quad	.LBB73
	.quad	.LBE73-.LBB73
	.byte	0x93
	.byte	0x13
	.long	0x715
	.uleb128 0x5
	.long	0xacd
	.uleb128 0x3
	.byte	0x77
	.sleb128 176
	.byte	0
	.uleb128 0xb
	.long	0xa1b
	.quad	.LBB75
	.quad	.LBE75-.LBB75
	.byte	0x94
	.byte	0x13
	.long	0x743
	.uleb128 0x5
	.long	0xa38
	.uleb128 0x3
	.byte	0x77
	.sleb128 800
	.uleb128 0x5
	.long	0xa2b
	.uleb128 0x3
	.byte	0x77
	.sleb128 768
	.byte	0
	.uleb128 0x1c
	.long	0xa64
	.quad	.LBB77
	.quad	.LBE77-.LBB77
	.byte	0x95
	.uleb128 0x5
	.long	0xa7f
	.uleb128 0x3
	.byte	0x77
	.sleb128 736
	.uleb128 0x5
	.long	0xa73
	.uleb128 0x3
	.byte	0x77
	.sleb128 704
	.byte	0
	.byte	0
	.uleb128 0x2
	.quad	.LBB79
	.quad	.LBE79-.LBB79
	.long	0x792
	.uleb128 0x1
	.string	"__Y"
	.byte	0x99
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 192
	.byte	0
	.uleb128 0x2
	.quad	.LBB80
	.quad	.LBE80-.LBB80
	.long	0x7b7
	.uleb128 0x1
	.string	"__Y"
	.byte	0x9b
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 208
	.byte	0
	.uleb128 0x2
	.quad	.LBB81
	.quad	.LBE81-.LBB81
	.long	0x7dc
	.uleb128 0x1
	.string	"__Y"
	.byte	0x9d
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 224
	.byte	0
	.uleb128 0x2
	.quad	.LBB82
	.quad	.LBE82-.LBB82
	.long	0x801
	.uleb128 0x1
	.string	"__Y"
	.byte	0x9f
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 240
	.byte	0
	.uleb128 0x2
	.quad	.LBB83
	.quad	.LBE83-.LBB83
	.long	0x826
	.uleb128 0x1
	.string	"__Y"
	.byte	0xa1
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 256
	.byte	0
	.uleb128 0x2
	.quad	.LBB84
	.quad	.LBE84-.LBB84
	.long	0x84b
	.uleb128 0x1
	.string	"__Y"
	.byte	0xa3
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 272
	.byte	0
	.uleb128 0x2
	.quad	.LBB85
	.quad	.LBE85-.LBB85
	.long	0x870
	.uleb128 0x1
	.string	"__Y"
	.byte	0xa5
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 288
	.byte	0
	.uleb128 0x2
	.quad	.LBB86
	.quad	.LBE86-.LBB86
	.long	0x895
	.uleb128 0x1
	.string	"__Y"
	.byte	0xa7
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 304
	.byte	0
	.uleb128 0x2
	.quad	.LBB87
	.quad	.LBE87-.LBB87
	.long	0x8ba
	.uleb128 0x1
	.string	"__Y"
	.byte	0xa9
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 320
	.byte	0
	.uleb128 0x2
	.quad	.LBB88
	.quad	.LBE88-.LBB88
	.long	0x8df
	.uleb128 0x1
	.string	"__Y"
	.byte	0xab
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 336
	.byte	0
	.uleb128 0x2
	.quad	.LBB89
	.quad	.LBE89-.LBB89
	.long	0x904
	.uleb128 0x1
	.string	"__Y"
	.byte	0xad
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 352
	.byte	0
	.uleb128 0x2
	.quad	.LBB90
	.quad	.LBE90-.LBB90
	.long	0x929
	.uleb128 0x1
	.string	"__Y"
	.byte	0xaf
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 368
	.byte	0
	.uleb128 0x2
	.quad	.LBB91
	.quad	.LBE91-.LBB91
	.long	0x94e
	.uleb128 0x1
	.string	"__Y"
	.byte	0xb1
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 384
	.byte	0
	.uleb128 0x2
	.quad	.LBB92
	.quad	.LBE92-.LBB92
	.long	0x973
	.uleb128 0x1
	.string	"__Y"
	.byte	0xb3
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 400
	.byte	0
	.uleb128 0x2
	.quad	.LBB93
	.quad	.LBE93-.LBB93
	.long	0x998
	.uleb128 0x1
	.string	"__Y"
	.byte	0xb5
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 416
	.byte	0
	.uleb128 0x2
	.quad	.LBB94
	.quad	.LBE94-.LBB94
	.long	0x9bd
	.uleb128 0x1
	.string	"__Y"
	.byte	0xb7
	.byte	0xd
	.long	0x120
	.uleb128 0x3
	.byte	0x77
	.sleb128 432
	.byte	0
	.uleb128 0x25
	.long	0xa8c
	.quad	.LBB56
	.quad	.LBE56-.LBB56
	.byte	0x1
	.byte	0x85
	.byte	0x17
	.byte	0
	.byte	0
	.uleb128 0xa
	.quad	.LBB95
	.quad	.LBE95-.LBB95
	.uleb128 0x1
	.string	"i"
	.byte	0xbb
	.byte	0xe
	.long	0x70
	.uleb128 0x3
	.byte	0x77
	.sleb128 116
	.byte	0
	.byte	0
	.uleb128 0x6
	.long	0xb5
	.uleb128 0x6
	.long	0x70
	.uleb128 0x6
	.long	0x370
	.uleb128 0x6
	.long	0x385
	.uleb128 0x9
	.long	0xa9
	.long	0xa1b
	.uleb128 0x10
	.long	0x43
	.byte	0xc7
	.byte	0
	.uleb128 0x11
	.long	.LASF48
	.byte	0x4
	.value	0x229
	.long	0x19d
	.long	0xa46
	.uleb128 0xe
	.string	"__A"
	.byte	0x4
	.value	0x229
	.byte	0x1d
	.long	0x19d
	.uleb128 0xe
	.string	"__B"
	.byte	0x4
	.value	0x229
	.byte	0x2a
	.long	0x19d
	.byte	0
	.uleb128 0x11
	.long	.LASF49
	.byte	0x4
	.value	0x1e2
	.long	0x19d
	.long	0xa64
	.uleb128 0xe
	.string	"__X"
	.byte	0x4
	.value	0x1e2
	.byte	0x1f
	.long	0x120
	.byte	0
	.uleb128 0x1d
	.long	.LASF50
	.byte	0x4
	.byte	0x9a
	.long	0x19d
	.long	0xa8c
	.uleb128 0x18
	.string	"__A"
	.byte	0x4
	.byte	0x9a
	.byte	0x1c
	.long	0x19d
	.uleb128 0x18
	.string	"__B"
	.byte	0x4
	.byte	0x9a
	.byte	0x29
	.long	0x19d
	.byte	0
	.uleb128 0x26
	.long	.LASF58
	.byte	0x2
	.value	0x4da
	.byte	0x1
	.long	0x19d
	.byte	0x3
	.uleb128 0x11
	.long	.LASF51
	.byte	0x2
	.value	0x3f1
	.long	0x19d
	.long	0xab8
	.uleb128 0xe
	.string	"__P"
	.byte	0x2
	.value	0x3f1
	.byte	0x24
	.long	0xab8
	.byte	0
	.uleb128 0x6
	.long	0x1a9
	.uleb128 0x11
	.long	.LASF52
	.byte	0x2
	.value	0x393
	.long	0x19d
	.long	0xadb
	.uleb128 0xe
	.string	"__P"
	.byte	0x2
	.value	0x393
	.byte	0x23
	.long	0xab8
	.byte	0
	.uleb128 0x1d
	.long	.LASF53
	.byte	0x3
	.byte	0x6f
	.long	0x120
	.long	0xaf7
	.uleb128 0x18
	.string	"__P"
	.byte	0x3
	.byte	0x6f
	.byte	0x21
	.long	0xaf7
	.byte	0
	.uleb128 0x6
	.long	0x12c
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0x21
	.sleb128 8
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x2107
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x21
	.byte	0
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x57
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0x21
	.sleb128 3
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0x26
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x13
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0x21
	.sleb128 9999999
	.byte	0
	.byte	0
	.uleb128 0x14
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x88
	.uleb128 0x21
	.sleb128 32
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x15
	.uleb128 0x5
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x16
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x17
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0x21
	.sleb128 100
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x18
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x19
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1a
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0x21
	.sleb128 310
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x1b
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 14
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x88
	.uleb128 0x21
	.sleb128 32
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x1c
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x57
	.uleb128 0x21
	.sleb128 19
	.byte	0
	.byte	0
	.uleb128 0x1d
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0x21
	.sleb128 3
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1e
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x1f
	.uleb128 0x1b
	.uleb128 0x1f
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x1f
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x20
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x21
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x22
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x23
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x24
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x25
	.uleb128 0x1d
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x57
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x26
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF27:
	.string	"__v16hi"
.LASF34:
	.string	"mul_u8"
.LASF58:
	.string	"_mm256_setzero_si256"
.LASF32:
	.string	"argc"
.LASF7:
	.string	"short int"
.LASF16:
	.string	"size_t"
.LASF30:
	.string	"malloc"
.LASF28:
	.string	"__v16hu"
.LASF56:
	.string	"main"
.LASF18:
	.string	"long double"
.LASF10:
	.string	"__uint32_t"
.LASF9:
	.string	"__uint16_t"
.LASF26:
	.string	"__v8si"
.LASF40:
	.string	"exp_u8"
.LASF37:
	.string	"arr_u16"
.LASF13:
	.string	"uint8_t"
.LASF22:
	.string	"__v8hi"
.LASF55:
	.string	"free"
.LASF29:
	.string	"__m256i"
.LASF20:
	.string	"float"
.LASF54:
	.ascii	"GNU C17 11.3.0 -march=znver2 -mmmx -mpopcnt -msse -msse2 -ms"
	.ascii	"se3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -msse4a -mno-fma4"
	.ascii	" -mno-xop -mfma -mno-avx512f -mbmi -mbmi2 -maes -mpclmul -mn"
	.ascii	"o-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-av"
	.ascii	"x512er -mno-avx512pf -mno-avx512vbmi -mno-avx512ifma -mno-av"
	.ascii	"x5124vnniw -mno-avx5124fmaps -mno-avx512vpopcntdq -mno-avx51"
	.ascii	"2vbmi2 -mno-gfni -mno-vpclmulqdq -mno-avx512vnni -mno-avx512"
	.ascii	"bitalg -mno-avx512bf16 -mno-avx512vp2intersect -mno-3dnow -m"
	.ascii	"adx -mabm -mno-cldemote -mclflushopt -mclwb -mclzero -mcx16 "
	.ascii	"-mno-enqcmd -mf16c -mfsgsbase -mfxsr -mno-hle -msahf -mno-lw"
	.ascii	"p -mlzcnt -mmovbe -mno-movdir64b -mno-movdiri -mno-mwaitx -m"
	.ascii	"no-pconfig -mno-pku -mno-prefetchwt1 -mprfchw -mno-ptwrite -"
	.ascii	"mrdpid -mrdrnd -mrdseed -mno-rtm -mno-serialize -mno-sgx -ms"
	.ascii	"ha -mno-shstk -mno-tbm -mno-tsxldtrk -mno-vaes -mno-waitpkg "
	.ascii	"-mwbnoinvd -mxsave -mxsavec -mxsaveopt -mxsaves -mno-amx-til"
	.ascii	"e -mno-amx-int8 -mno-a"
	.string	"mx-bf16 -mno-uintr -mno-hreset -mno-kl -mno-widekl -mno-avxvnni --param=l1-cache-size=32 --param=l1-cache-line-size=64 --param=l2-cache-size=512 -mtune=znver2 -g -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection"
.LASF17:
	.string	"long long int"
.LASF11:
	.string	"long int"
.LASF31:
	.string	"printf"
.LASF52:
	.string	"_mm256_load_si256"
.LASF8:
	.string	"__uint8_t"
.LASF43:
	.string	"out_1"
.LASF44:
	.string	"out_2"
.LASF45:
	.string	"out_3"
.LASF46:
	.string	"out_4"
.LASF2:
	.string	"unsigned char"
.LASF49:
	.string	"_mm256_cvtepu8_epi16"
.LASF48:
	.string	"_mm256_mullo_epi16"
.LASF6:
	.string	"signed char"
.LASF19:
	.string	"long long unsigned int"
.LASF15:
	.string	"uint32_t"
.LASF4:
	.string	"unsigned int"
.LASF14:
	.string	"uint16_t"
.LASF33:
	.string	"argv"
.LASF36:
	.string	"mul_u16"
.LASF3:
	.string	"short unsigned int"
.LASF42:
	.string	"out_c"
.LASF35:
	.string	"arr_u8"
.LASF38:
	.string	"n_u8"
.LASF12:
	.string	"char"
.LASF41:
	.string	"exp_u16"
.LASF25:
	.string	"__v4di"
.LASF24:
	.string	"__m128i"
.LASF5:
	.string	"long unsigned int"
.LASF21:
	.string	"double"
.LASF23:
	.string	"__v16qi"
.LASF50:
	.string	"_mm256_adds_epu16"
.LASF57:
	.string	"sum_inline_u16"
.LASF53:
	.string	"_mm_lddqu_si128"
.LASF47:
	.string	"load"
.LASF51:
	.string	"_mm256_lddqu_si256"
.LASF39:
	.string	"n_u16"
	.section	.debug_line_str,"MS",@progbits,1
.LASF1:
	.string	"/mnt/ext/projects/pasteur/src/pasteur/marginal/native"
.LASF0:
	.string	"impl.c"
	.ident	"GCC: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
