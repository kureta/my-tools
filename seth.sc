(
// create a Buffer placeholder
b = Buffer.alloc(s, 1, 1);

// receive alloc message
OSCdef(\pyAlloc, { |msg|
    var id = msg[1], size = msg[2];
    b.free;  // free any old
    b = Buffer.alloc(s, size, 1);
    ("Allocated buffer " ++ id ++ " size " ++ size).postln;
}, '/pythonBuffer/alloc');

// receive chunks and write them
OSCdef(\pySet, { |msg|
    var id = msg[1], offset = msg[2], data;
	data = msg.copyRange(3, msg.size-1);
    b.setn(offset, data);
}, '/pythonBuffer/setn');

// Synth for playing buffer
SynthDef(\player, { |out=0, bufnum, rate=1.0, amp=0.5|
    var sig = PlayBuf.ar(1, bufnum, rate, loop: 1);
    Out.ar(out, sig * amp);
}).add;
)

a = Synth(\player);

s.options.maxLogins = 4;
s.boot;
s.reboot;
s.quit;